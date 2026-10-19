"""Compare real LIBERO data and optionally full-model loss for first-frame decoding."""
import argparse
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
from pathlib import Path
import time

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import default_collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--uint8', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = OmegaConf.load('configs/libero/train_action.yaml')
    # The component loader resolves relative checkpoints against model_id.
    # Match train_8h200.sh, which supplies repository-absolute paths.
    for key in ('model_id', 'dit_checkpoint_path', 'action_dit_pretrained_path'):
        cfg.model[key] = str(Path(cfg.model[key]).resolve())
        if not Path(cfg.model[key]).exists():
            raise FileNotFoundError(cfg.model[key])
    cfg.data.train.decode_only_first_video_frame = False
    cfg.data.train.processor.num_image_obs_steps = 65
    reference = instantiate(cfg.data.train)
    optimized_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    optimized_cfg.decode_only_first_video_frame = True
    optimized_cfg.processor.num_image_obs_steps = 1
    optimized = instantiate(optimized_cfg)
    rows, pair = [], None
    # Cover all four suites, including dataset and episode boundary padding.
    for index in [0, 100, 53228, 53229, 120537, 120538, 173432, 173433, 277712]:
        t = time.perf_counter()
        os.environ.pop("VPP2_VIDEO_DECODE_UINT8", None)
        full = reference[index]
        full_s = time.perf_counter() - t
        t = time.perf_counter()
        if args.uint8:
            os.environ["VPP2_VIDEO_DECODE_UINT8"] = "1"
        short = optimized[index]
        short_s = time.perf_counter() - t
        assert full.keys() == short.keys()
        for key in full:
            expected = full[key]
            if key == 'video':
                expected = expected[:, :1]
            elif key == 'image_is_pad':
                expected = expected[:1]
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(short[key], expected, rtol=0, atol=0)
            else:
                assert short[key] == expected, key
        assert short['video_num_frames'] == 17
        rows.append(dict(index=index, reference_seconds=full_s, first_frame_seconds=short_s))
        if index == 100:
            pair = (full, short)
    report = dict(status='PASS', real_data_exact_equal=True, uint8_decode=args.uint8, samples=rows)
    if args.gpu:
        from vpp2.libero.trainer import LiberoTrainer
        model = instantiate(cfg.model, model_dtype=torch.bfloat16, device='cuda')
        trainer = LiberoTrainer.__new__(LiberoTrainer)
        trainer.train_mode = 'action_only'
        trainer._apply_train_mode(model)
        def evaluate(sample):
            torch.manual_seed(123)
            torch.cuda.manual_seed_all(123)
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, _ = model(default_collate([sample]))
            loss.backward()
            return (loss.detach().float().cpu(), {
                name: parameter.grad.detach().cpu().clone()
                for name, parameter in model.named_parameters() if parameter.requires_grad
            })

        def compare(a, b):
            delta2 = ref2 = 0.0
            max_abs = 0.0
            different = 0
            for name in a[1]:
                av, bv = a[1][name].float(), b[1][name].float()
                difference = av - bv
                delta2 += float(difference.double().square().sum())
                ref2 += float(av.double().square().sum())
                max_abs = max(max_abs, float(difference.abs().max()))
                different += int(not torch.equal(av, bv))
            return dict(loss_abs_diff=abs(float(a[0])-float(b[0])),
                        gradient_relative_l2=(delta2/max(ref2, 1e-30))**0.5,
                        gradient_max_abs=max_abs, differing_gradient_tensors=different)

        full, short = pair
        # Diagnose backend repeatability before judging the data-path change.
        first = evaluate(full)
        repeat = evaluate(full)
        report['native_same_input_repeat'] = compare(first, repeat)
        del repeat
        changed = evaluate(short)
        report['native_first_frame_comparison'] = compare(first, changed)
        del first, changed
        report['status'] = 'DETERMINISTIC_CHECK_PENDING'
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
        # Test exact equivalence with deterministic kernels, without relaxing tolerance.
        # This is a separate validation process; production training is unchanged.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        first = evaluate(full)
        changed = evaluate(short)
        report['deterministic_first_frame_comparison'] = compare(first, changed)
        report['status'] = 'CHECKING'
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
        torch.testing.assert_close(first[0], changed[0], rtol=0, atol=0)
        for name in first[1]:
            torch.testing.assert_close(first[1][name], changed[1][name], rtol=0, atol=0)
        report.update(status='PASS', full_model_loss=float(first[0]), full_model_loss_exact_equal=True,
                      deterministic_trainable_gradients_exact_equal=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
