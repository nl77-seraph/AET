"""Minimal correctness check for the complete AET-WF model path."""

from __future__ import annotations

import inspect
import io
from unittest.mock import patch
import sys
from pathlib import Path

import pytest
import torch


from aet_wf.models import AETWFModel, BackgroundAwareUOT, IndependentLocalPooling, PrototypeBank
from aet_wf.predict import _build_model, _infer


@pytest.mark.parametrize("aggregation_mode", ["uot", "independent_local"])
def test_aet_forward_backward_and_prototype_state(aggregation_mode) -> None:
    torch.manual_seed(3)
    model = AETWFModel(
        5,
        num_class_prototypes=3,
        num_background_prototypes=4,
        dropout=0,
        uot_iters=8,
        aggregation_mode=aggregation_mode,
    )
    assert list(inspect.signature(AETWFModel.forward).parameters) == ["self", "features", "lengths"]

    features = torch.randn(2, 512, 2)
    lengths = torch.tensor([512, 32])
    model.eval()
    torch.testing.assert_close(
        model(features, lengths)["logits"], model(features.transpose(1, 2), lengths)["logits"]
    )
    model.train()
    output = model(features, lengths)
    batch, steps = output["token_mask"].shape
    assert output["logits"].shape == (2, 5)
    assert output["transport_plan"].shape == (batch, steps, 6)
    assert output["evidence_mass"].shape == (2, 5)
    assert output["evidence_quality"].shape == (2, 5)
    assert output["evidence_vectors"].shape == (2, 5, 256)
    assert bool((~output["token_mask"][1]).any())
    assert torch.count_nonzero(output["transport_plan"][~output["token_mask"]]) == 0
    for key in ("logits", "transport_plan", "evidence_mass", "evidence_quality", "evidence_vectors"):
        assert bool(torch.isfinite(output[key]).all()), key

    output["logits"].square().mean().backward()
    for parameter in (
        model.encoder.backbone.block1_conv1.weight,
        model.prototype_bank.class_values,
        model.prototype_bank.background_values,
        model.presence_head.semantic[-1].weight,
        model.presence_head.transport[-1].weight,
    ):
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())

    bank = PrototypeBank(5, 3, 4)
    bank.load_state_dict(model.prototype_bank.state_dict())
    torch.testing.assert_close(bank.class_values, model.prototype_bank.class_values)
    assert not any(key.startswith("affinity.prototype_bank") for key in model.state_dict())
    bank.normalize_()
    torch.testing.assert_close(
        bank.class_values.norm(dim=-1), torch.ones(5, 3), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        bank.background_values.norm(dim=-1), torch.ones(4), atol=1e-6, rtol=0
    )


def test_uot_padding_and_cpu_gpu_tolerance() -> None:
    torch.manual_seed(7)
    solver = BackgroundAwareUOT(num_iters=12)
    affinity = torch.randn(2, 9, 6).clamp(-1, 1)
    mask = torch.tensor([[1] * 9, [1] * 5 + [0] * 4], dtype=torch.bool)
    cpu = solver(affinity.masked_fill(~mask.unsqueeze(-1), -torch.inf), mask)
    assert bool(torch.isfinite(cpu).all())
    assert torch.count_nonzero(cpu[~mask]) == 0

    if torch.cuda.is_available():
        gpu = solver.cuda()(
            affinity.cuda().masked_fill(~mask.cuda().unsqueeze(-1), -torch.inf), mask.cuda()
        ).cpu()
        torch.testing.assert_close(cpu, gpu, atol=2e-5, rtol=2e-4)


def test_rejects_all_padding() -> None:
    model = AETWFModel(3, dropout=0, uot_iters=2)
    with pytest.raises(ValueError, match="positive valid length"):
        model(torch.zeros(1, 128, 2), torch.tensor([0]))


def test_local_manual_aggregation_independence_and_background() -> None:
    model = AETWFModel(2, dropout=0, uot_epsilon=0.1, aggregation_mode="independent_local").eval()
    tokens = torch.zeros(1, 3, 256)
    tokens[0, 0, 0], tokens[0, 1, 1] = 1, 2
    mask = torch.tensor([[True, True, False]])
    affinity = torch.tensor([[[0.0, 0.2, -0.2], [0.2, -0.2, 0.6], [-torch.inf] * 3]])
    def forward(value):
        with patch.object(model.encoder, "forward", return_value={"tokens": tokens, "token_mask": mask}), patch.object(model.affinity, "forward", return_value=value):
            return model(torch.zeros(1, 2, 512), torch.tensor([512]))
    output = forward(affinity)
    gates = torch.sigmoid(torch.tensor([[1.0, -1.0], [-2.0, 2.0]]))
    weights = gates / 2
    mass = weights.sum(0)
    quality = (weights * affinity[0, :2, 1:]).sum(0) / mass
    vectors = weights.T @ tokens[0, :2] / mass[:, None]
    torch.testing.assert_close(output["evidence_mass"][0], mass)
    torch.testing.assert_close(output["evidence_quality"][0], quality)
    torch.testing.assert_close(output["evidence_vectors"][0], vectors)
    torch.testing.assert_close(output["background_mass"], ((1 - gates.max(-1).values) / 2).sum().reshape(1))
    assert float(output["transport_plan"].sum()) > 1  # Independent support is not a joint probability distribution.
    stronger = affinity.clone()
    stronger[:, :2, 1] += 0.2
    assert forward(stronger)["evidence_mass"][0, 0] > mass[0]
    other_class = affinity.clone()
    other_class[:, :2, 2] += 0.2
    changed = forward(other_class)
    for key in ("evidence_mass", "evidence_quality", "evidence_vectors", "logits"):
        torch.testing.assert_close(changed[key][:, 0], output[key][:, 0], atol=0, rtol=0)
    stronger_background = affinity.clone()
    stronger_background[:, :2, 0] += 0.3
    assert bool((forward(stronger_background)["evidence_mass"] < output["evidence_mass"]).all())


def test_local_padding_clipping_and_finite_gradients() -> None:
    solver = IndependentLocalPooling(0.1)
    # Includes out-of-range affinity and both very weak/strong class support.
    affinity = torch.tensor([[[2.0, -2.0, 0.5], [-0.5, 0.5, -0.5], [-torch.inf] * 3]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    plan = solver(affinity, mask)
    torch.testing.assert_close(plan[0, 0, 1], torch.sigmoid(torch.tensor(-10.0)) / 2)
    assert torch.count_nonzero(plan[~mask]) == 0 and bool(torch.isfinite(plan).all())
    short = solver(affinity[:, :2], mask[:, :2])
    torch.testing.assert_close(plan[:, :2], short, atol=0, rtol=0)
    plan.square().sum().backward()
    assert bool(torch.isfinite(affinity.grad).all())
    assert torch.count_nonzero(affinity.grad[~mask]) == 0
    assert torch.count_nonzero(affinity.grad[:, :2]) > 0
    assert not solver(affinity, mask, detach_plan=True).requires_grad
    with pytest.raises(ValueError, match="valid token"):
        solver(torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="shape mismatch"):
        solver(torch.zeros(1, 2, 3), torch.ones(1, 3, dtype=torch.bool))
    with pytest.raises(FloatingPointError):
        solver(torch.full((1, 1, 2), torch.nan), torch.ones(1, 1, dtype=torch.bool))
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            IndependentLocalPooling(invalid)
    for kwargs in ({"background_prior": 1}, {"uot_rho_token": 0}, {"uot_iters": 0}, {"aggregation_mode": "wrong"}):
        with pytest.raises(ValueError):
            AETWFModel(2, **({"aggregation_mode": "independent_local"} | kwargs))


def test_local_paired_initialization_warmup_and_checkpoint_prediction() -> None:
    args = dict(num_classes=3, dropout=0, uot_iters=3)
    torch.manual_seed(123)
    uot = AETWFModel(**args)
    uot_rng = torch.get_rng_state()
    torch.manual_seed(123)
    local = AETWFModel(**args, aggregation_mode="independent_local")
    assert torch.equal(uot_rng, torch.get_rng_state())
    assert dict(uot.named_parameters()).keys() == dict(local.named_parameters()).keys()
    assert uot.state_dict().keys() == local.state_dict().keys()
    for name, value in uot.state_dict().items():
        torch.testing.assert_close(value, local.state_dict()[name], atol=0, rtol=0)
    features = torch.randn(2, 2, 512)
    lengths = torch.tensor([512, 31])
    for detached in (True, False):
        local.detach_plan = detached
        local.zero_grad(set_to_none=True)
        local(features, lengths)["logits"].square().mean().backward()
        assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in local.parameters())
        assert (torch.count_nonzero(local.prototype_bank.background_values.grad) == 0) == detached
    for model, model_args in ((uot, args), (local, args | {"aggregation_mode": "independent_local"})):
        model.eval()
        buffer = io.BytesIO()
        torch.save({"method": "aet", "model_args": model_args, "model": model.state_dict()}, buffer)
        buffer.seek(0)
        restored = _build_model(torch.load(buffer, weights_only=False)).eval()
        assert restored.aggregation_mode == model.aggregation_mode
        assert isinstance(restored.transport, type(model.transport))
        expected = model(features, lengths)["logits"]
        torch.testing.assert_close(restored(features, lengths)["logits"], expected, atol=0, rtol=0)
        logits, ids = _infer(restored, "aet", [{"features": features, "lengths": lengths, "sample_ids": torch.tensor([8, 3])}], torch.device("cpu"), "float32")
        torch.testing.assert_close(torch.from_numpy(logits), expected, atol=0, rtol=0)
        assert ids.tolist() == [8, 3]
