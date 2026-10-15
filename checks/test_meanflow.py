import copy

import pytest
import torch
from torch import nn

from vpp2.meanflow.model import MeanFlowLibero, derivative_mode, prepare_condition
from vpp2.meanflow.objectives import *
from vpp2.meanflow.teacher import rollout
from vpp2.models.wan21_14b.action_dit import ActionDiT
from vpp2.models.wan21_14b.mot import MoT
from vpp2.models.wan21_14b.vpp2 import VPP2
from vpp2.models.wan21_14b.wan_video_dit import WanVideoDiT


def tiny_model(cls=VPP2, layers=2, device="cpu", dtype=torch.float32, checkpoint=False):
    video = WanVideoDiT(
        hidden_dim=24,
        ffn_dim=48,
        freq_dim=8,
        text_dim=16,
        num_heads=2,
        attn_head_dim=8,
        num_layers=layers,
        use_gradient_checkpointing=checkpoint,
        video_attention_mask_mode="bidirectional",
    )
    action = ActionDiT(
        hidden_dim=16,
        action_dim=3,
        ffn_dim=32,
        text_dim=16,
        freq_dim=8,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=8,
        num_layers=layers,
        use_gradient_checkpointing=checkpoint,
    )
    mot = MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=checkpoint)
    vae = nn.Identity()
    vae.temporal_downsample_factor = 4
    model = cls(
        video,
        action,
        mot,
        vae,
        text_dim=16,
        proprio_dim=3,
        device=device,
        torch_dtype=dtype,
        action_visible_video_frames=2,
        video_train_sampling_strategy="pure_noise",
    )
    model.mot.to(dtype=dtype)
    return model


def inputs(model, batch=2):
    d, dt = model.device, model.torch_dtype
    context = torch.randn(batch, 3, 16, device=d, dtype=dt)
    mask = torch.ones(batch, 3, device=d, dtype=torch.bool)
    context, mask = model._append_proprio_to_context(
        context, mask, torch.randn(batch, 3, device=d, dtype=dt)
    )
    return dict(
        context=context,
        context_mask=mask,
        action=torch.randn(batch, 4, 3, device=d, dtype=dt),
        condition_latents=torch.randn(batch, 16, 1, 2, 2, device=d, dtype=dt),
        clip_fea=None,
        fuse_vae_embedding_in_latents=False,
    )


def test_constant_teacher_and_endpoint_gradient():
    x = torch.randn(3, 4, 2)
    v = torch.randn_like(x)
    sigmas = torch.tensor([1.0, 0.75, 0.2, 0.0])
    states = torch.stack([x - (1 - s) * v for s in sigmas])
    z, r, t, target = teacher_intervals(states, sigmas, 1.0)
    torch.testing.assert_close(target, v)
    torch.testing.assert_close(z - target, states[-1])
    prediction = (v + 0.3).requires_grad_()
    masked_mse(prediction, target).backward()
    assert prediction.grad.norm() > 0


def test_masks_and_empty():
    p = torch.ones(2, 3, 4, requires_grad=True)
    target = torch.zeros_like(p)
    pad = torch.tensor([[False, True, True], [True, True, True]])
    loss = masked_mse(p, target, pad)
    assert loss.item() == 1
    loss.backward()
    assert p.grad[1].count_nonzero() == 0
    assert p.grad[0, 1:].count_nonzero() == 0
    assert masked_mse(p, target, torch.ones_like(pad)).item() == 0
    with pytest.raises(ValueError):
        masked_mse(p, target[:, :, :1], pad)


def test_jvp_scale_and_boundary_fd():
    z = torch.randn(3, 2, 2)
    r = torch.tensor([0.0, 0.1, 0.8])
    t = torch.tensor([0.0, 0.5, 1.0])
    v = torch.randn_like(z)
    fn = lambda z, r, t: 2 * z + expand_time(1000 * t, z)
    exact = 2 * v + 1000
    torch.testing.assert_close(directional_derivative(fn, z, r, t, v), exact)
    torch.testing.assert_close(
        directional_derivative(fn, z, r, t, v, "finite_difference", 0.01),
        exact,
        rtol=2e-4,
        atol=0.01,
    )


def test_generator_policy_gradients():
    p = nn.Parameter(torch.tensor(0.7))
    gen = SplitGenerator()
    clean = torch.randn(3, 4, 2)
    noise = torch.randn_like(clean)
    r = torch.zeros(3)
    t = torch.tensor([0.5, 0.7, 0.9])
    predict = lambda z, r, t: p * z

    def derivative(z, r, t, v, rows):
        return (v[rows] * (1 + t[rows, None, None])).detach()

    loss, metrics = data_loss(
        predict, derivative, clean, noise, r, t, objective="kmf", generator=gen
    )
    assert metrics["loss_generator"] > 0
    loss.backward()
    assert p.grad.abs() > 0
    assert any(x.grad is not None and x.grad.norm() > 0 for x in gen.parameters())
    # A detached policy-only target cannot update generator parameters.
    gen.zero_grad()
    p.grad = None
    loss, _ = data_loss(
        predict, derivative, clean, noise, r, t, objective="kmf", generator=gen, alpha=0.0
    )
    loss.backward()
    assert all(x.grad is None or x.grad.count_nonzero() == 0 for x in gen.parameters())


def test_diagonal_no_derivatives():
    clean = torch.randn(2, 4, 3)
    noise = torch.randn_like(clean)
    t = torch.rand(2)

    def bad(*args):
        raise AssertionError("Diagonal must not run JVP")

    loss, _ = data_loss(lambda z, r, t: noise - clean, bad, clean, noise, t, t)
    assert loss.item() == 0


def test_zero_end_branch_and_strict_loading():
    m = tiny_model()
    i = inputs(m)
    n = torch.randn(2, 16, 2, 2, 2)
    c, _ = prepare_condition(m, i, n, joint=True)
    z = i["action"]
    t = torch.tensor([500.0, 750.0])
    baseline = m.predict_action_velocity_with_cache(z, t, **c)
    m.action_expert.enable_meanflow()
    actual = m.predict_action_velocity_with_cache(
        z, t, timestep_end=torch.tensor([0.0, 250.0]), **c
    )
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    actual.square().mean().backward()
    assert m.action_expert.time_end_embedding[-1].weight.grad.norm() > 0
    broken = dict(m.action_expert.state_dict())
    broken.pop("head.weight")
    with pytest.raises(RuntimeError):
        m.action_expert.load_state_dict(broken, strict=True)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_joint_cache_forward_and_gradient_equivalence(checkpoint):
    torch.manual_seed(13)
    m = tiny_model(checkpoint=checkpoint)
    m.train()
    i = inputs(m)
    n = torch.randn(2, 16, 2, 2, 2)
    vt = torch.full((2,), 1000.0)
    at = torch.tensor([300.0, 600.0])
    vp = m.video_expert.pre_dit(
        n, vt, i["context"], i["context_mask"], condition_latents=i["condition_latents"]
    )
    ap = m.action_expert.pre_dit(i["action"], at, i["context"], i["context_mask"])
    ns = vp["tokens"].shape[1]
    mask = m._build_mot_attention_mask(ns, 4, 1, "cpu")
    assert not mask[:ns, ns:].any()
    out = m.mot(
        embeds_all={"video": vp["tokens"], "action": ap["tokens"]},
        attention_mask=mask,
        freqs_all={"video": vp["freqs"], "action": ap["freqs"]},
        context_all={
            "video": {"context": vp["context"], "mask": vp["context_mask"]},
            "action": {"context": ap["context"], "mask": ap["context_mask"]},
        },
        t_mod_all={"video": vp["t_mod"], "action": ap["t_mod"]},
    )
    pv = m.video_expert.post_dit(out["video"], vp)
    pa = m.action_expert.post_dit(out["action"], ap)
    params = list(m.parameters())
    loss = pv.square().mean() + pa.square().mean()
    grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
    c, cv = prepare_condition(m, i, n, joint=True)
    ca = m.predict_action_velocity_with_cache(i["action"], at, **c)
    torch.testing.assert_close(ca, pa, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(cv, pv, rtol=2e-5, atol=2e-6)
    cg = torch.autograd.grad(cv.square().mean() + ca.square().mean(), params, allow_unused=True)
    for a, b in zip(grads, cg):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-6)


def test_frozen_cache_direct_proprio_gradient():
    m = tiny_model(MeanFlowLibero)
    m.video_expert.requires_grad_(False)
    i = inputs(m)
    c, _ = prepare_condition(m, i, torch.randn(2, 16, 2, 2, 2), joint=False)
    assert not c["video_kv_cache"][0]["k"].requires_grad
    m.predict_action_velocity_with_cache(
        i["action"], torch.ones(2) * 500, **c
    ).square().mean().backward()
    assert all(p.grad is None for p in m.video_expert.parameters())
    assert m.proprio_encoder.weight.grad.norm() > 0


def test_actual_conditioned_jvp():
    m = tiny_model()
    m.action_expert.enable_meanflow()
    i = inputs(m)
    c, _ = prepare_condition(m, i, torch.randn(2, 16, 2, 2, 2), joint=False)
    z = i["action"]
    r = torch.zeros(2)
    t = torch.tensor([0.4, 0.7])
    v = torch.randn_like(z)
    fn = lambda z, r, t: m.predict_action_velocity_with_cache(
        z, t * 1000, timestep_end=r * 1000, **c
    )
    with derivative_mode(m):
        jvp = directional_derivative(fn, z, r, t, v)
        fd = directional_derivative(fn, z, r, t, v, "finite_difference", 0.0001)
    assert torch.isfinite(jvp).all()
    torch.testing.assert_close(jvp, fd, rtol=0.04, atol=0.01)


def test_cached_teacher_matches_scheduler():
    m = tiny_model().eval()
    i = inputs(m)
    c, _ = prepare_condition(m, i, torch.randn(2, 16, 2, 2, 2))
    states, sigmas = rollout(m, i["action"], c, 10, 5.0)
    assert sigmas[0] == 1 and sigmas[-1] == 0 and (sigmas[:-1] > sigmas[1:]).all()
    z = i["action"]
    times, deltas = m.infer_action_scheduler.build_inference_schedule(10, "cpu", z.dtype, 5.0)
    with torch.no_grad():
        for t, d in zip(times, deltas):
            z = m.infer_action_scheduler.step(
                m._predict_action_noise_with_cache(z, t.expand(2), **c), d, z
            )
    torch.testing.assert_close(z, states[-1], rtol=0, atol=0)


def test_inference_counts_all_40_blocks_once():

    m = tiny_model(layers=40).eval()
    m.action_expert.enable_meanflow()
    m.action_sampler = "meanflow"
    m._encode_input_image_latents_tensor = lambda **kwargs: torch.zeros(1, 16, 1, 2, 2)
    m._encode_clip_fea_from_image = lambda x: None
    counts = [0] * 40
    calls = {"action": 0, "video": 0}
    # MoT calls block internals directly, so instrument each block's unique self-attention Q.
    handles = []
    for index, block in enumerate(m.action_expert.blocks):

        def hook(module, args, out, index=index):
            counts[index] += 1

        handles.append(block.self_attn.q.register_forward_hook(hook))
    orig = m.mot.forward_action_with_video_cache
    prefill = m.mot.prefill_video_cache

    def counted(*args, **kwargs):
        calls["action"] += 1
        return orig(*args, **kwargs)

    def counted_video(*args, **kwargs):
        calls["video"] += 1
        return prefill(*args, **kwargs)

    m.mot.forward_action_with_video_cache = counted
    m.mot.prefill_video_cache = counted_video
    result = m.infer_action(
        None,
        torch.zeros(1, 3, 16, 16),
        4,
        context=torch.zeros(1, 3, 16),
        context_mask=torch.ones(1, 3, dtype=torch.bool),
        num_inference_steps=1,
        seed=1,
    )
    assert result["action"].shape == (4, 3)
    assert counts == [1] * 40 and calls == {"action": 1, "video": 1}
    for h in handles:
        h.remove()
    with pytest.raises(ValueError):
        m.infer_action(None, torch.zeros(1, 3, 16, 16), 4, num_inference_steps=10)


def test_checkpoint_pair_loading_and_mismatch(tmp_path):
    from vpp2.checkpoint_compat import ACTION_FORMAT
    from vpp2.meanflow.runtime import load_fm

    m = tiny_model()
    original = copy.deepcopy(m.action_expert.state_dict())
    action = dict(
        format=ACTION_FORMAT,
        step=100000,
        resolved_config={},
        action_expert=original,
        proprio_encoder=m.proprio_encoder.state_dict(),
    )
    video = dict(dit=m.video_expert.state_dict(), step=100000)
    ap = tmp_path / "action.pt"
    vp = tmp_path / "video.pt"
    torch.save(action, ap)
    torch.save(video, vp)
    spec = dict(format="deployment_pair", action_checkpoint=str(ap), video_checkpoint=str(vp))
    target = tiny_model()
    provenance = load_fm(target, spec, "robodojo")
    assert provenance["source_step"] == 100000 and len(provenance["action_sha256"]) == 64
    for key, value in original.items():
        torch.testing.assert_close(target.action_expert.state_dict()[key], value)
    video["step"] = 10000
    torch.save(video, vp)
    with pytest.raises(ValueError, match="matched joint pair"):
        load_fm(target, spec, "robodojo")
    video["step"] = 100000
    torch.save(video, vp)
    del action["action_expert"]["head.weight"]
    torch.save(action, ap)
    with pytest.raises(RuntimeError):
        load_fm(target, spec, "robodojo")


def test_terminal_padding_and_future_independence():
    m = tiny_model()

    def encode(video, **kwargs):
        # Deterministic per-time surrogate for this input-boundary test only.
        small = video.mean((1, 3, 4), keepdim=True)[:, :, ::4]
        return small.expand(-1, 16, -1, 2, 2).contiguous()

    m._encode_video_latents = encode
    sample = dict(
        video=torch.randn(2, 3, 5, 16, 16),
        condition_video=torch.randn(2, 3, 1, 16, 16),
        context=torch.randn(2, 3, 16),
        context_mask=torch.ones(2, 3, dtype=torch.bool),
        proprio=torch.randn(2, 1, 3),
        action=torch.randn(2, 4, 3),
        action_is_pad=torch.tensor([[False, True, True, True]] * 2),
        action_terminal_mask=torch.tensor([[False, True, False, False]] * 2),
        clip_fea=torch.zeros(2, 257, 1280),
    )
    first = m.build_inputs(sample, training_stage="joint")
    assert first["action_loss_is_pad"].tolist() == [[False, False, True, True]] * 2
    noise = torch.randn_like(first["input_latents"])
    c1, _ = prepare_condition(m, first, noise)
    sample["video"][:, :, 1:] += 100
    second = m.build_inputs(sample, training_stage="joint")
    c2, _ = prepare_condition(m, second, noise)
    for a, b in zip(c1["video_kv_cache"], c2["video_kv_cache"]):
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_libero_new_update_metadata():
    from omegaconf import OmegaConf

    from vpp2.libero.protocols import validate_config

    cfg = OmegaConf.load("configs/libero/eval_standard.yaml")
    validate_config(cfg)
    cfg.action_sampler = "meanflow"
    cfg.action_nfe = 1
    cfg.action_step = 20
    cfg.source_action_step = 30000
    validate_config(cfg)
    cfg.source_action_step = 20
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_meanflow_export_keeps_end_time_and_actual_update(tmp_path):
    from omegaconf import OmegaConf
    from vpp2.meanflow.model import MeanFlowVPP2
    from vpp2.weights import export

    model = tiny_model(MeanFlowVPP2)
    model.configure_meanflow(
        dict(objective="kmf", split="learned"),
        dict(student=dict(source_step=100000)),
    )
    with torch.no_grad():
        model.action_expert.time_end_embedding[-1].weight.fill_(0.1)
    source = tmp_path / "student.pt"
    model.save_checkpoint(source, step=20)
    cfg = OmegaConf.load("configs/robodojo/train_100k.yaml")
    cfg.meanflow = dict(objective="kmf")
    cfg.model.meanflow = "${meanflow}"
    cfg.model.benchmark = "robodojo"
    config = tmp_path / "config.yaml"
    OmegaConf.save(cfg, config)
    stats = tmp_path / "stats.json"
    stats.write_text("{}")
    out = tmp_path / "bundle"
    manifest = export(source, config, stats, out, 20)
    payload = torch.load(out / "action.pt", weights_only=True)
    assert manifest["step"] == 20 and manifest["inference"]["steps"] == 1
    assert payload["meanflow"]["student_update_step"] == 20
    assert payload["meanflow"]["provenance"]["student"]["source_step"] == 100000
    assert "split_generator" not in payload and "teacher" not in payload
    architecture = payload["resolved_config"]["model"]
    assert architecture["action_dit_config"]["meanflow"]
    assert "meanflow" not in architecture and "benchmark" not in architecture
    restored = tiny_model()
    restored.action_expert.enable_meanflow()
    restored.action_expert.load_state_dict(payload["action_expert"], strict=True)
    for key, value in model.action_expert.state_dict().items():
        torch.testing.assert_close(restored.action_expert.state_dict()[key], value, rtol=0, atol=0)


def test_fm_continuation_provenance_survives_resume(tmp_path):
    from omegaconf import OmegaConf
    from vpp2.checkpoint_compat import configure_fm_continuation
    from vpp2.libero.protocols import validate_config

    model = tiny_model()
    checkpoint = tmp_path / "fm.pt"
    model.save_checkpoint(checkpoint, step=100000)
    cfg = OmegaConf.create(
        dict(
            fm_continuation=True,
            resume=None,
            resume_ckpt=str(checkpoint),
            output_dir=str(tmp_path / "run"),
            train_mode="joint",
        )
    )
    configure_fm_continuation(model, cfg)
    student = tmp_path / "b2.pt"
    model.save_checkpoint(student, step=20)
    source = torch.load(student, weights_only=True)["fm_continuation"]
    assert source["source_step"] == 100000 and source["student_update_step"] == 20
    cfg.resume = str(tmp_path / "run/checkpoints/state/step_000020")
    cfg.resume_ckpt = None
    resumed = tiny_model()
    configure_fm_continuation(resumed, cfg)
    assert resumed.fm_continuation_source == model.fm_continuation_source
    evaluation = OmegaConf.load("configs/libero/eval_standard.yaml")
    evaluation.fm_continuation = True
    evaluation.source_action_step = 30000
    evaluation.action_step = 20
    evaluation.action_nfe = 1
    validate_config(evaluation)
    evaluation.source_action_step = 20
    with pytest.raises(ValueError):
        validate_config(evaluation)
