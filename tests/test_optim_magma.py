import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.optim import _build_magma_factors, muon_step_fused


def test_magma_alignment_controls_scale():
    # Block 0: perfectly aligned; Block 1: perfectly anti-aligned.
    grads = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    momentum = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    scale_ema = torch.ones(2, dtype=torch.float32)

    updated_scale, mask = _build_magma_factors(
        momentum,
        grads,
        scale_ema,
        p=1.0,
        tau=1.0,
        ema_beta=0.0,
    )

    assert updated_scale[0] > updated_scale[1]
    assert mask.all()


def test_magma_mask_probability_extremes():
    grads = torch.randn(4, 2, 2)
    momentum = torch.randn(4, 2, 2)

    scale_ema = torch.ones(4, dtype=torch.float32)
    _, mask0 = _build_magma_factors(momentum, grads, scale_ema, p=0.0, tau=2.0, ema_beta=0.9)
    assert not mask0.any()

    scale_ema = torch.ones(4, dtype=torch.float32)
    _, mask1 = _build_magma_factors(momentum, grads, scale_ema, p=1.0, tau=2.0, ema_beta=0.9)
    assert mask1.all()


def test_magma_ema_update_rule():
    grads = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    momentum = grads.clone()  # cosine similarity == 1
    scale_ema = torch.tensor([0.2], dtype=torch.float32)

    updated_scale, _ = _build_magma_factors(
        momentum,
        grads,
        scale_ema,
        p=1.0,
        tau=1.0,
        ema_beta=0.9,
    )

    expected = 0.9 * 0.2 + 0.1 * torch.sigmoid(torch.tensor(1.0))
    assert torch.allclose(updated_scale, expected)


def test_setup_optimizer_includes_magma_fields():
    config = GPTConfig(
        sequence_len=32,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()

    optimizer = model.setup_optimizer(magma=True, magma_p=0.5, magma_tau=2.0, magma_ema_beta=0.9)
    muon_groups = [group for group in optimizer.param_groups if group["kind"] == "muon"]

    assert muon_groups
    for group in muon_groups:
        assert group["magma"] is True
        assert group["magma_p"] == 0.5
        assert group["magma_tau"] == 2.0
        assert group["magma_ema_beta"] == 0.9


def test_muon_step_masks_only_gradient_not_weight_decay():
    stacked_grads = torch.ones(1, 2, 2, dtype=torch.float32)
    stacked_params = torch.ones(1, 2, 2, dtype=torch.float32)
    momentum_buffer = torch.zeros_like(stacked_grads)
    second_momentum_buffer = torch.zeros(1, 2, 1, dtype=torch.float32)

    muon_step_fused(
        stacked_grads,
        stacked_params,
        momentum_buffer,
        second_momentum_buffer,
        torch.tensor(0.0, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(0.1, dtype=torch.float32),
        torch.tensor(0.0, dtype=torch.float32),
        torch.ones(1, 1, 1, dtype=torch.float32),
        torch.zeros(1, 1, 1, dtype=torch.float32),
        0,
        -1,
    )

    # MAGMA mask zeroes gradient-driven update, but cautious weight decay should still apply.
    assert torch.allclose(stacked_params, torch.full_like(stacked_params, 0.9))


def test_magma_disabled_reuses_identity_buffers():
    config = GPTConfig(
        sequence_len=32,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()

    optimizer = model.setup_optimizer(magma=False)
    muon_groups = [group for group in optimizer.param_groups if group["kind"] == "muon"]
    assert muon_groups

    for group in muon_groups:
        for p in group["params"]:
            p.grad = torch.ones_like(p)
    optimizer.step()

    first_group = muon_groups[0]
    state = optimizer.state[first_group["params"][0]]
    assert "magma_identity_scale" in state
    assert "magma_identity_mask" in state
    scale_id = id(state["magma_identity_scale"])
    mask_id = id(state["magma_identity_mask"])

    for group in muon_groups:
        for p in group["params"]:
            p.grad = torch.ones_like(p)
    optimizer.step()

    state = optimizer.state[first_group["params"][0]]
    assert id(state["magma_identity_scale"]) == scale_id
    assert id(state["magma_identity_mask"]) == mask_id
