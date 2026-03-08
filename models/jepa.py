"""Teacher-student JEPA model for hidden-state readout learning."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.span_sampler import ContiguousSpanSampler


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False
    module.eval()


def _update_ema(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    if not 0.0 <= decay <= 1.0:
        raise ValueError(f"EMA decay must be in [0,1], got {decay}")

    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
        for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
            if teacher_buffer.dtype.is_floating_point:
                teacher_buffer.data.mul_(decay).add_(student_buffer.data, alpha=1.0 - decay)
            else:
                teacher_buffer.data.copy_(student_buffer.data)


class TeacherStudentJEPA(nn.Module):
    """JEPA objective on pretrained hidden states.

    Student gets a masked/corrupted view; teacher sees full hidden states.
    Target is a pooled teacher span representation projected to D_out.
    """

    def __init__(
        self,
        student_readout: nn.Module,
        d_in: int,
        d_out: int,
        predictor_hidden_dim: int = 1024,
        use_span_position: bool = True,
        ema_decay: float = 0.99,
        span_sampler: ContiguousSpanSampler | None = None,
    ) -> None:
        super().__init__()
        self.student_readout = student_readout
        self.teacher_readout = copy.deepcopy(student_readout)
        _freeze_module(self.teacher_readout)

        self.student_span_projector = nn.Linear(d_in, d_out)
        # Target span projection is fixed in the current objective: the predictor learns
        # to match a frozen projected teacher span, so this module should not be trainable.
        _freeze_module(self.student_span_projector)
        self.teacher_span_projector = copy.deepcopy(self.student_span_projector)
        _freeze_module(self.teacher_span_projector)

        self.use_span_position = use_span_position
        self.ema_decay = ema_decay
        self.span_sampler = span_sampler or ContiguousSpanSampler()

        self.position_encoder: nn.Module | None
        if use_span_position:
            self.position_encoder = nn.Sequential(
                nn.Linear(3, d_out),
                nn.GELU(),
                nn.Linear(d_out, d_out),
            )
            predictor_input_dim = 2 * d_out
        else:
            self.position_encoder = None
            predictor_input_dim = d_out

        self.predictor = nn.Sequential(
            nn.LayerNorm(predictor_input_dim),
            nn.Linear(predictor_input_dim, predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(predictor_hidden_dim, d_out),
        )

    def update_teacher(self, decay: float | None = None) -> None:
        """Applies EMA update to the teacher readout."""
        ema_decay = self.ema_decay if decay is None else decay
        _update_ema(self.student_readout, self.teacher_readout, ema_decay)

    def forward(
        self,
        full_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        student_hidden_states: torch.Tensor | None = None,
        spans: torch.Tensor | None = None,
        return_teacher_output: bool = False,
    ) -> dict[str, Any]:
        """Computes JEPA student prediction and teacher target."""
        if full_hidden_states.ndim != 3:
            raise ValueError(
                f"Expected full_hidden_states [B,L,D], got {tuple(full_hidden_states.shape)}"
            )
        if attention_mask.shape != full_hidden_states.shape[:2]:
            raise ValueError(
                f"Expected attention_mask [B,L], got {tuple(attention_mask.shape)}"
            )
        if student_hidden_states is None:
            student_hidden_states = full_hidden_states
        if student_hidden_states.shape != full_hidden_states.shape:
            raise ValueError("student_hidden_states shape must match full_hidden_states shape")

        if spans is None:
            spans = self.span_sampler.sample(attention_mask=attention_mask)
        if spans.shape != (full_hidden_states.shape[0], 2):
            raise ValueError(f"Expected spans [B,2], got {tuple(spans.shape)}")

        student_out = self.student_readout(
            hidden_states=student_hidden_states,
            attention_mask=attention_mask,
        )

        teacher_out: dict[str, torch.Tensor] | None = None
        with torch.no_grad():
            if return_teacher_output:
                teacher_out = self.teacher_readout(
                    hidden_states=full_hidden_states,
                    attention_mask=attention_mask,
                )
            target = self._compute_teacher_span_target(
                hidden_states=full_hidden_states,
                attention_mask=attention_mask,
                spans=spans,
            )

        predictor_input = [student_out["z_seq"]]
        if self.position_encoder is not None:
            span_position = self._encode_span_position(
                spans=spans,
                attention_mask=attention_mask,
            )
            predictor_input.append(span_position)
        pred_input = torch.cat(predictor_input, dim=-1)
        pred = self.predictor(pred_input)

        return {
            "pred": pred,
            "target": target,
            "spans": spans,
            "z_seq_student": student_out["z_seq"],
            "z_seq_teacher": teacher_out["z_seq"] if teacher_out is not None else None,
            "student_output": student_out,
            "teacher_output": teacher_out,
        }

    def _compute_teacher_span_target(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        spans: torch.Tensor,
    ) -> torch.Tensor:
        span_mask = ContiguousSpanSampler.spans_to_mask(
            spans=spans, seq_length=hidden_states.shape[1]
        )
        span_mask = span_mask & attention_mask.bool()
        mask = span_mask.unsqueeze(-1).to(hidden_states.dtype)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled_span = (hidden_states * mask).sum(dim=1) / counts
        projected_target = self.teacher_span_projector(pooled_span)
        projected_target = F.normalize(projected_target, p=2, dim=-1)
        return projected_target

    def _encode_span_position(self, spans: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        valid_lengths = attention_mask.sum(dim=1).clamp(min=1).to(spans.dtype)
        starts = spans[:, 0].to(torch.float32)
        ends = spans[:, 1].to(torch.float32)
        lengths = (spans[:, 1] - spans[:, 0]).to(torch.float32)
        norm = valid_lengths.to(torch.float32)
        features = torch.stack([starts / norm, ends / norm, lengths / norm], dim=-1)
        return self.position_encoder(features)
