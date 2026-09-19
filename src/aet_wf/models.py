"""Prototype-guided transport for multi-tab website fingerprinting."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# DFNet is copied from the TMWF reference implementation.
# Original source header: Author: lok; Edited By: jzx-bupt.
# Only this convolutional backbone is included; see README.md.

class DFNet(nn.Module):
    def __init__(self, dropout):
        super(DFNet, self).__init__()

        # Block1
        filter_num = [0, 32, 64, 128, 256]
        kernel_size = [0, 8, 8, 8, 8]
        conv_stride_size = [0, 1, 1, 1, 1]
        pool_stride_size = [0, 4, 4, 4, 4]
        pool_size = [0, 8, 8, 8, 8]

        self.block1_conv1 = nn.Conv1d(in_channels=1, out_channels=filter_num[1],
                                      kernel_size=kernel_size[1],
                                      stride=conv_stride_size[1], padding=kernel_size[1] // 2)
        self.block1_bn1 = nn.BatchNorm1d(num_features=filter_num[1])
        self.block1_elu1 = nn.ELU(alpha=1.0)
        self.block1_conv2 = nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[1], kernel_size=kernel_size[1],
                                      stride=conv_stride_size[1], padding=kernel_size[1] // 2)
        self.block1_bn2 = nn.BatchNorm1d(num_features=filter_num[1])
        self.block1_elu2 = nn.ELU(alpha=1.0)
        self.block1_pool = nn.MaxPool1d(kernel_size=pool_size[1], stride=pool_stride_size[1], padding=pool_size[1] // 2)
        self.block1_dropout = nn.Dropout(p=dropout)

        self.block2_conv1 = nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[2], kernel_size=kernel_size[2],
                                      stride=conv_stride_size[2], padding=kernel_size[2] // 2)
        self.block2_bn1 = nn.BatchNorm1d(num_features=filter_num[2])
        self.block2_relu1 = nn.ReLU()
        self.block2_conv2 = nn.Conv1d(in_channels=filter_num[2], out_channels=filter_num[2], kernel_size=kernel_size[2],
                                      stride=conv_stride_size[2], padding=kernel_size[2] // 2)
        self.block2_bn2 = nn.BatchNorm1d(num_features=filter_num[2])
        self.block2_relu2 = nn.ReLU()
        self.block2_pool = nn.MaxPool1d(kernel_size=pool_size[2], stride=pool_stride_size[2], padding=pool_size[2] // 2)
        self.block2_dropout = nn.Dropout(p=dropout)

        self.block3_conv1 = nn.Conv1d(in_channels=filter_num[2], out_channels=filter_num[3], kernel_size=kernel_size[3],
                                      stride=conv_stride_size[3], padding=kernel_size[3] // 2)
        self.block3_bn1 = nn.BatchNorm1d(num_features=filter_num[3])
        self.block3_relu1 = nn.ReLU()
        self.block3_conv2 = nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[3], kernel_size=kernel_size[3],
                                      stride=conv_stride_size[3], padding=kernel_size[3] // 2)
        self.block3_bn2 = nn.BatchNorm1d(num_features=filter_num[3])
        self.block3_relu2 = nn.ReLU()
        self.block3_pool = nn.MaxPool1d(kernel_size=pool_size[3], stride=pool_stride_size[3], padding=pool_size[3] // 2)
        self.block3_dropout = nn.Dropout(p=dropout)

        self.block4_conv1 = nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[4], kernel_size=kernel_size[4],
                                      stride=conv_stride_size[4], padding=kernel_size[4] // 2)
        self.block4_bn1 = nn.BatchNorm1d(num_features=filter_num[4])
        self.block4_relu1 = nn.ReLU()
        self.block4_conv2 = nn.Conv1d(in_channels=filter_num[4], out_channels=filter_num[4], kernel_size=kernel_size[4],
                                      stride=conv_stride_size[4], padding=kernel_size[4] // 2)
        self.block4_bn2 = nn.BatchNorm1d(num_features=filter_num[4])
        self.block4_relu2 = nn.ReLU()
        self.block4_pool = nn.MaxPool1d(kernel_size=pool_size[4], stride=pool_stride_size[4], padding=pool_size[4] // 2)
        self.block4_dropout = nn.Dropout(p=dropout)

    def forward(self, input):

        if len(input.shape) == 2:
            x = input.unsqueeze(1)
        else:
            x = input

        # Block 1
        x = self.block1_conv1(x)
        x = self.block1_bn1(x)
        x = self.block1_elu1(x)
        x = self.block1_conv2(x)
        x = self.block1_bn2(x)
        x = self.block1_elu2(x)
        x = self.block1_pool(x)
        x = self.block1_dropout(x)

        # Block 2
        x = self.block2_conv1(x)
        x = self.block2_bn1(x)
        x = self.block2_relu1(x)
        x = self.block2_conv2(x)
        x = self.block2_bn2(x)
        x = self.block2_relu2(x)
        x = self.block2_pool(x)
        x = self.block2_dropout(x)

        # Block 3
        x = self.block3_conv1(x)
        x = self.block3_bn1(x)
        x = self.block3_relu1(x)
        x = self.block3_conv2(x)
        x = self.block3_bn2(x)
        x = self.block3_relu2(x)
        x = self.block3_pool(x)
        x = self.block3_dropout(x)

        # Block 4
        x = self.block4_conv1(x)
        x = self.block4_bn1(x)
        x = self.block4_relu1(x)
        x = self.block4_conv2(x)
        x = self.block4_bn2(x)
        x = self.block4_relu2(x)
        x = self.block4_pool(x)
        x = self.block4_dropout(x)
        return x.transpose(1, 2)


class MultiScaleTrafficEncoder(nn.Module):
    """A shared two-channel DF-style encoder that preserves local tokens."""

    output_dim = 256

    def __init__(self, input_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        if input_dim != 2:
            raise ValueError("PGT expects direction and log-IAT channels")
        self.input_dim = input_dim
        self.backbone = DFNet(dropout)
        old = self.backbone.block1_conv1
        self.backbone.block1_conv1 = nn.Conv1d(
            input_dim,
            old.out_channels,
            old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        self.norm = nn.LayerNorm(self.output_dim)

    @staticmethod
    def token_lengths(lengths: Tensor) -> Tensor:
        """Propagate lengths through DFNet's four conv/pool blocks."""
        result = lengths.long()
        for _ in range(4):
            result = (result + 2) // 4 + 1
        return result

    def forward(self, features: Tensor, lengths: Tensor) -> dict[str, Tensor]:
        if features.ndim != 3:
            raise ValueError("expected a three-dimensional feature tensor")
        if features.shape[-1] == self.input_dim:
            batch, sequence_length, _ = features.shape
            channel_first = features.transpose(1, 2)
        elif features.shape[1] == self.input_dim:
            batch, _, sequence_length = features.shape
            channel_first = features
        else:
            raise ValueError(f"expected [batch,length,{self.input_dim}] or [batch,{self.input_dim},length]")
        lengths = torch.as_tensor(lengths, device=features.device)
        if lengths.shape != (batch,) or bool(((lengths < 1) | (lengths > sequence_length)).any()):
            raise ValueError("lengths must contain one positive valid length per sample")

        packet_mask = torch.arange(sequence_length, device=features.device)[None] < lengths[:, None]
        masked = channel_first * packet_mask.unsqueeze(1).to(features.dtype)
        tokens = self.norm(self.backbone(masked))
        valid_steps = self.token_lengths(lengths).clamp(max=tokens.shape[1])
        token_mask = torch.arange(tokens.shape[1], device=features.device)[None] < valid_steps[:, None]
        tokens = tokens.masked_fill(~token_mask.unsqueeze(-1), 0)
        summary = tokens.sum(1) / token_mask.sum(1, keepdim=True).clamp_min(1)
        return {"tokens": tokens, "token_mask": token_mask, "global_summary": summary}


class PrototypeBank(nn.Module):
    """Trainable local class and background prototypes stored in state_dict."""

    def __init__(
        self,
        num_classes: int,
        num_class_prototypes: int,
        num_background_prototypes: int,
        dim: int = 256,
    ):
        super().__init__()
        if min(num_classes, num_class_prototypes, num_background_prototypes, dim) < 1:
            raise ValueError("prototype dimensions must be positive")
        self.num_classes = int(num_classes)
        self.num_class_prototypes = int(num_class_prototypes)
        self.num_background_prototypes = int(num_background_prototypes)
        self.dim = int(dim)
        self.class_values = nn.Parameter(
            F.normalize(torch.randn(num_classes, num_class_prototypes, dim), dim=-1)
        )
        self.background_values = nn.Parameter(
            F.normalize(torch.randn(num_background_prototypes, dim), dim=-1)
        )

    def class_prototypes(self) -> Tensor:
        return F.normalize(self.class_values, dim=-1)

    def background_prototypes(self) -> Tensor:
        return F.normalize(self.background_values, dim=-1)

    def class_summaries(self) -> Tensor:
        return F.normalize(self.class_prototypes().mean(1), dim=-1)

    @torch.no_grad()
    def normalize_(self) -> None:
        self.class_values.copy_(F.normalize(self.class_values, dim=-1))
        self.background_values.copy_(F.normalize(self.background_values, dim=-1))


class EvidenceAffinity(nn.Module):
    """Cosine token-to-prototype affinity with log-mean-exp aggregation."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        if temperature < 1e-3:
            raise ValueError("temperature must be at least 1e-3")
        self.temperature = float(temperature)

    def forward(
        self,
        tokens: Tensor,
        token_mask: Tensor,
        class_prototypes: Tensor,
        background_prototypes: Tensor,
    ) -> Tensor:
        if tokens.ndim != 3 or token_mask.shape != tokens.shape[:2]:
            raise ValueError("tokens/token_mask shape mismatch")
        if class_prototypes.ndim != 3 or background_prototypes.ndim != 2:
            raise ValueError("expected class [C,R,D] and background [R0,D] prototypes")
        if class_prototypes.shape[-1] != tokens.shape[-1] or background_prototypes.shape[-1] != tokens.shape[-1]:
            raise ValueError("token/prototype dimensions do not match")
        normalized = F.normalize(tokens, dim=-1)
        class_similarity = torch.einsum(
            "btd,crd->btcr", normalized, class_prototypes
        )
        background_similarity = torch.einsum(
            "btd,rd->btr", normalized, background_prototypes
        )
        tau = self.temperature
        class_affinity = tau * (
            torch.logsumexp(class_similarity / tau, dim=-1)
            - math.log(class_prototypes.shape[1])
        )
        background_affinity = tau * (
            torch.logsumexp(background_similarity / tau, dim=-1)
            - math.log(background_prototypes.shape[0])
        )
        affinity = torch.cat((background_affinity.unsqueeze(-1), class_affinity), dim=-1)
        return affinity.masked_fill(~token_mask.unsqueeze(-1), -torch.inf)


class BackgroundAwareUOT(nn.Module):
    """Batched log-domain unbalanced Sinkhorn with background in column zero."""

    def __init__(
        self,
        epsilon: float = 0.05,
        rho_token: float = 0.5,
        rho_class: float = 0.5,
        num_iters: int = 30,
        background_prior: float = 0.5,
    ):
        super().__init__()
        if min(epsilon, rho_token, rho_class) <= 0 or num_iters < 1:
            raise ValueError("UOT regularization and iterations must be positive")
        if not 0 < background_prior < 1:
            raise ValueError("background_prior must be in (0,1)")
        self.epsilon = float(epsilon)
        self.rho_token = float(rho_token)
        self.rho_class = float(rho_class)
        self.num_iters = int(num_iters)
        self.background_prior = float(background_prior)

    def forward(self, affinity: Tensor, token_mask: Tensor, detach_plan: bool = False) -> Tensor:
        if affinity.ndim != 3 or token_mask.shape != affinity.shape[:2]:
            raise ValueError("affinity/token_mask shape mismatch")
        if affinity.shape[-1] < 2 or bool((~token_mask.any(1)).any()):
            raise ValueError("UOT needs at least one class and one valid token per sample")

        affinity = affinity.float()
        token_mask = token_mask.bool()
        classes = affinity.shape[-1] - 1
        # All--inf rows make logsumexp gradients NaN even if their forward
        # values are masked later. Use a finite placeholder during the solver;
        # padding is still excluded by its dual and zeroed in the final plan.
        safe_affinity = affinity.masked_fill(~token_mask.unsqueeze(-1), 0)
        unit_affinity = ((safe_affinity + 1.0) * 0.5).clamp(0.0, 1.0)
        log_kernel = -(1.0 - unit_affinity) / self.epsilon
        log_kernel = log_kernel.masked_fill(~token_mask.unsqueeze(-1), 0)

        counts = token_mask.sum(1, keepdim=True).float()
        log_supply = (-counts.log()).expand_as(token_mask).masked_fill(~token_mask, -torch.inf)
        prior = affinity.new_full((classes + 1,), (1.0 - self.background_prior) / classes)
        prior[0] = self.background_prior
        log_prior = prior.log().unsqueeze(0)
        alpha = self.rho_token / (self.rho_token + self.epsilon)
        beta = self.rho_class / (self.rho_class + self.epsilon)
        log_left = affinity.new_zeros(token_mask.shape)
        log_right = affinity.new_zeros((affinity.shape[0], classes + 1))

        for _ in range(self.num_iters):
            row_lse = torch.logsumexp(log_kernel + log_right.unsqueeze(1), dim=-1)
            row_lse = row_lse.masked_fill(~token_mask, 0)
            log_left = alpha * (log_supply.masked_fill(~token_mask, 0) - row_lse)
            log_left = log_left.masked_fill(~token_mask, -torch.inf)
            column_lse = torch.logsumexp(log_kernel + log_left.unsqueeze(-1), dim=1)
            log_right = beta * (log_prior - column_lse)

        plan = torch.exp(log_kernel + log_left.unsqueeze(-1) + log_right.unsqueeze(1))
        plan = plan.masked_fill(~token_mask.unsqueeze(-1), 0)
        if not bool(torch.isfinite(plan).all()):
            raise FloatingPointError("non-finite UOT plan")
        return plan.detach() if detach_plan else plan


class IndependentLocalPooling(nn.Module):
    """Independent class-versus-background weights, without joint transport."""

    def __init__(self, epsilon: float = 0.05):
        super().__init__()
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("local pooling epsilon must be finite and positive")
        self.epsilon = float(epsilon)

    def forward(self, affinity: Tensor, token_mask: Tensor, detach_plan: bool = False) -> Tensor:
        if affinity.ndim != 3 or token_mask.shape != affinity.shape[:2]:
            raise ValueError("affinity/token_mask shape mismatch")
        token_mask = token_mask.bool()
        if affinity.shape[-1] < 2 or bool((~token_mask.any(1)).any()):
            raise ValueError("local pooling needs a class and a valid token per sample")
        safe_affinity = affinity.float().masked_fill(~token_mask.unsqueeze(-1), 0)
        unit_affinity = ((safe_affinity + 1.0) * 0.5).clamp(0.0, 1.0)
        log_kernel = -(1.0 - unit_affinity) / self.epsilon
        gates = torch.sigmoid(log_kernel[:, :, 1:] - log_kernel[:, :, :1])
        supply = token_mask.float() / token_mask.sum(1, keepdim=True)
        class_weights = supply.unsqueeze(-1) * gates
        # Background is diagnostic only; it never rescales the class columns.
        background = supply * (1.0 - gates.max(-1).values)
        plan = torch.cat((background.unsqueeze(-1), class_weights), dim=-1)
        if not bool(torch.isfinite(plan).all()):
            raise FloatingPointError("non-finite local pooling weights")
        return plan.detach() if detach_plan else plan


class SharedPresenceHead(nn.Module):
    """One presence MLP shared by every website class."""

    def __init__(
        self,
        dim: int = 256,
        dropout: float = 0.1,
        expected_class_mass: float = 1.0,
    ):
        super().__init__()
        if expected_class_mass <= 0:
            raise ValueError("expected class mass must be positive")
        self.register_buffer("expected_class_mass", torch.tensor(float(expected_class_mass)))
        self.semantic = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
        )
        self.transport = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(
        self,
        evidence_vectors: Tensor,
        class_summaries: Tensor,
        evidence_mass: Tensor,
        evidence_quality: Tensor,
    ) -> Tensor:
        summaries = class_summaries.unsqueeze(0).expand(evidence_vectors.shape[0], -1, -1)
        normalized_evidence = F.normalize(evidence_vectors, dim=-1)
        interaction = normalized_evidence * summaries
        cosine = interaction.sum(-1)
        relative_log_mass = torch.log(
            evidence_mass.clamp_min(1e-8) / self.expected_class_mass
        )
        transport_features = torch.stack(
            (relative_log_mass, evidence_quality, cosine), dim=-1
        )
        return (
            self.semantic(interaction).squeeze(-1)
            + self.transport(transport_features).squeeze(-1)
        )


class AETWFModel(nn.Module):
    """Recognize a label set by transporting local evidence to anchored classes."""

    def __init__(
        self,
        num_classes: int,
        *,
        num_class_prototypes: int = 8,
        num_background_prototypes: int = 16,
        dropout: float = 0.1,
        affinity_temperature: float = 0.1,
        uot_epsilon: float = 0.05,
        uot_rho_token: float = 0.5,
        uot_rho_class: float = 0.03,
        uot_iters: int = 30,
        background_prior: float = 0.5,
        detach_plan: bool = False,
        aggregation_mode: str = "uot",
    ):
        super().__init__()
        if aggregation_mode not in {"uot", "independent_local"}:
            raise ValueError("aggregation_mode must be uot or independent_local")
        self.aggregation_mode = aggregation_mode
        self.encoder = MultiScaleTrafficEncoder(input_dim=2, dropout=dropout)
        self.prototype_bank = PrototypeBank(
            num_classes,
            num_class_prototypes,
            num_background_prototypes,
            self.encoder.output_dim,
        )
        self.affinity = EvidenceAffinity(affinity_temperature)
        self.transport = BackgroundAwareUOT(
            uot_epsilon,
            uot_rho_token,
            uot_rho_class,
            uot_iters,
            background_prior,
        )
        if aggregation_mode == "independent_local":
            # Keep the same recipe validation above, including inactive UOT fields.
            self.transport = IndependentLocalPooling(uot_epsilon)
        class_mass_prior = (1.0 - background_prior) / num_classes
        self.presence_head = SharedPresenceHead(
            self.encoder.output_dim, dropout, class_mass_prior
        )
        self.detach_plan = bool(detach_plan)

    def forward(self, features: Tensor, lengths: Tensor) -> dict[str, Tensor]:
        encoded = self.encoder(features, lengths)
        tokens, token_mask = encoded["tokens"], encoded["token_mask"]
        affinity = self.affinity(
            tokens,
            token_mask,
            self.prototype_bank.class_prototypes(),
            self.prototype_bank.background_prototypes(),
        )
        plan = self.transport(affinity, token_mask, self.detach_plan)
        class_plan = plan[:, :, 1:]
        class_affinity = affinity[:, :, 1:].masked_fill(~token_mask.unsqueeze(-1), 0)
        evidence_mass = class_plan.sum(1)
        denominator = evidence_mass.clamp_min(1e-8)
        evidence_quality = (class_plan * class_affinity).sum(1) / denominator
        evidence_vectors = torch.einsum("btc,btd->bcd", class_plan, tokens.float())
        evidence_vectors = evidence_vectors / denominator.unsqueeze(-1)
        logits = self.presence_head(
            evidence_vectors,
            self.prototype_bank.class_summaries(),
            evidence_mass,
            evidence_quality,
        )
        return {
            "logits": logits,
            "tokens": tokens,
            "token_mask": token_mask,
            "affinity": affinity,
            "transport_plan": plan,
            "evidence_mass": evidence_mass,
            "evidence_quality": evidence_quality,
            "evidence_vectors": evidence_vectors,
            "background_mass": plan[:, :, 0].sum(1),
        }
