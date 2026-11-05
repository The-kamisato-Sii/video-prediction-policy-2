"""Real-data/full-model checks for the isolated one-pass reconstruction ablation."""
import argparse
import gc
import json
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .data import collate_flow_action
from .model import FlowActionAdapters
from .runtime import load_config, create_dataset, create_model


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="runs/libero_flow_implementation/reconstruction/single")
    p.add_argument("--tokens", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--reconstruction-loss", choices=("l1", "mse"), default="l1")
    p.add_argument("--available-manifest", default="runs/libero_flow_implementation/storage_gpu_profile/available.json")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    from vpp2.utils import misc
    misc.register_work_dir(out)
    cfg = load_config(["flow.objective=direct_reconstruction", "flow.missing_motion=skip",
        f"flow.available_manifest={args.available_manifest}",
        f"flow.reconstruction_loss={args.reconstruction_loss}",
        "wandb.enabled=false"])
    torch.manual_seed(42)
    ds = create_dataset(cfg)
    samples = {i: ds[ds.first_index(i)] for i in (0, 1)}
    assert "motion" not in samples[0] and "action" not in samples[1]
    write(out / "data.json", dict(provenance={i:s["provenance"] for i,s in samples.items()},
        source_frames=samples[1]["motion_frame_index"].tolist(),
        dt4_endpoints=samples[1]["flow_target_frame_index"].tolist(),
        missing_labels_absent=True))

    def reject_motion_weights(path, *a, **kw):
        assert "geometry_motion_vae" not in str(path), "Loaded forbidden MotionVAE weights"
        return load_file(path, *a, **kw)
    print("Loading full shared model with RANDOM reconstruction decoder", flush=True)
    with patch("safetensors.torch.load_file", side_effect=reject_motion_weights):
        model = create_model(cfg)
    # Read only official metadata, not tensors, to independently check that ALL
    # codec module names/shapes/parameter counts match the original architecture.
    with safe_open(str(Path(cfg.flow.codec_path)/"diffusion_pytorch_model.safetensors"),
                   framework="pt", device="cpu") as f:
        official_shapes = {k: list(f.get_slice(k).get_shape()) for k in f.keys()}
    actual_shapes = {k:list(v.shape) for k,v in model.flow_codec.state_dict().items()}
    assert actual_shapes == official_shapes
    official_codec_count = sum(torch.Size(s).numel() for s in official_shapes.values())
    reference_common = sum(p.numel() for name,p in model.named_parameters()
        if not name.startswith(("flow_codec.", "flow_adapters.", "motion_queries")))
    write(out/"architecture.json", dict(codec_parameters=official_codec_count,
        codec_shapes_match_official_metadata=True, official_motion_weights_not_loaded=True,
        common_parameters=reference_common))

    for n in args.tokens:
        cfg.flow.tokens = n
        model.flow_adapters = FlowActionAdapters(model.action_expert.hidden_dim, n, motion_channels=model.flow_codec.latent_channels).to(model.device, model.torch_dtype)
        model.motion_queries = torch.nn.Parameter(torch.randn(1,n,512,device=model.device,dtype=model.torch_dtype)*.02)
        model.configure_trainability()
        model.last_shapes = {}
        reference_adapters = FlowActionAdapters(512, n, motion_channels=model.flow_codec.latent_channels)
        reference_count = reference_common + official_codec_count + sum(p.numel() for p in reference_adapters.parameters())
        total = sum(p.numel() for p in model.parameters())
        assert total - reference_count == n*512, (total, reference_count, n)
        del reference_adapters
        torch.cuda.reset_peak_memory_stats()
        report = dict(tokens=n, reconstruction_loss=model.reconstruction_loss,
            total_parameters=total, reference_parameters=reference_count,
            added_parameters=total-reference_count,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), branches=[])
        for indicator in (0,1):
            print(f"N={n}, indicator={indicator}: real forward/backward", flush=True)
            batch = collate_flow_action([samples[indicator]])
            model.zero_grad(set_to_none=True)
            with patch.object(model.flow_codec, "encode", side_effect=AssertionError("GT encoding forbidden")):
                if indicator:
                    with patch.object(model.train_action_scheduler, "add_noise", side_effect=AssertionError("Flow noising forbidden")):
                        loss, metrics = model(batch)
                else:
                    loss, metrics = model(batch)
                assert torch.isfinite(loss)
                loss.backward()
            watched = dict(backbone=model.action_expert.blocks[0].ffn[0].weight,
                queries=model.motion_queries, unpatch=model.flow_adapters.motion_unpatch.weight,
                geometry_decoder=model.flow_codec.decoder.conv_out.weight,
                flow_decoder=model.flow_codec.decoder_2.conv_out.weight,
                action_head=model.flow_adapters.action_head[-1].weight)
            gradients = {k:float(v.grad.float().norm()) if v.grad is not None else 0.0 for k,v in watched.items()}
            assert gradients["backbone"] > 0
            for key in (("queries","unpatch","geometry_decoder","flow_decoder") if indicator else ("action_head",)):
                assert gradients[key] > 0, (key, gradients)
            assert not any(p.grad is not None for p in model.video_expert.parameters())
            assert not any(p.grad is not None for p in model.flow_codec.encoder.parameters())
            row = dict(indicator=indicator, loss=float(loss.detach()), metrics=metrics,
                       gradients=gradients, shapes=dict(model.last_shapes[str(indicator)]))
            model.zero_grad(set_to_none=True)
            del loss
            gc.collect()
            model.eval()
            # Strip every GT field: inference can use only observations/conditions.
            inference = {k:batch[k] for k in ("video","video_num_frames","context","context_mask","proprio")}
            with patch.object(model.mot, "forward_action_with_video_cache",
                              wraps=model.mot.forward_action_with_video_cache) as forward:
                prediction = model.sample_task(inference, indicator=indicator, steps=1 if indicator else 10)
                assert forward.call_count == (1 if indicator else 10)
                row["inference_dit_calls"] = forward.call_count
            assert all(torch.isfinite(x).all() for x in prediction.values())
            row["sampled_shapes"] = {k:list(v.shape) for k,v in prediction.items()}
            if indicator:
                maps = torch.cat([prediction["point_map"],prediction["scene_flow"]],dim=2)
                valid = batch["motion_valid"].to(model.device)
                target = batch["motion"].to(model.device)
                errors = model.reconstruction_errors(maps,target,valid)
                altered = target.clone()
                altered[~valid.repeat_interleave(3,dim=2)] = 10000
                changed = model.reconstruction_errors(maps,altered,valid)
                assert all(torch.equal(a,b) for a,b in zip(errors,changed))
                empty = model.reconstruction_errors(maps,target,torch.zeros_like(valid))
                assert all(torch.equal(x,torch.zeros_like(x)) for x in empty)
                row["invalid_pixels_excluded"] = True
            report["branches"].append(row)
            print(json.dumps(row), flush=True)
            del prediction
            if indicator: del maps, target, altered, errors, changed, empty
            model.configure_trainability()
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
        report["status"] = "passed"
        write(out/f"model_n{n}.json",report)

    checkpoint = out / "checkpoint_roundtrip.pt"
    model.save_flow_checkpoint(checkpoint, step=7, resolved_cfg=cfg)
    probes = [model.motion_queries, model.flow_codec.decoder.conv_out.weight,
              model.flow_codec.decoder_2.conv_out.weight, model.flow_adapters.motion_unpatch.weight]
    expected = [p.detach().clone() for p in probes]
    with torch.no_grad():
        for parameter in probes: parameter.add_(1)
    assert model.load_checkpoint(checkpoint) == 7
    assert all(torch.equal(p,e) for p,e in zip(probes,expected))
    write(out/"checkpoint.json",dict(status="passed",step=7,random_decoder_and_queries_restored=True,
                                    checkpoint_bytes=checkpoint.stat().st_size))
    # This is a disposable random-model diagnostic, not a training checkpoint.
    checkpoint.unlink()
    write(out/"completion.json",dict(status="passed",tokens=args.tokens))


if __name__ == "__main__":
    main()
