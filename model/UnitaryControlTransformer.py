import math
from typing import Callable, Dict, Sequence, Tuple, Optional, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap   # <-- 1-line opt-in for vectorised maps

from tqdm import tqdm


__all__ = ["CompositePulseTransformerEncoder"]



###############################################################################
# Model with SCORE Embedding
###############################################################################

class UnitaryControlTransformer(nn.Module):
    """Transformer encoder mapping *U_target* → pulse sequence.

    Each pulse is a continuous vector of parameters whose ranges are supplied
    via ``pulse_space``.  For a single‑qubit drive this could be (Δ, Ω, φ, t).
    """

    def __init__(
        self,
        num_qubits: int,
        pulse_space: Dict[str, Tuple[float, float]],
        max_pulses: int = 16,
        d_model: int = 256,
        n_layers: int = 12,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        
        super().__init__()
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits

        # ------------------------------------------------------------------
        # Pulse parameter space
        # ------------------------------------------------------------------
        self.param_names: Sequence[str] = list(pulse_space.keys())
        self.param_ranges: torch.Tensor = torch.tensor(
            [pulse_space[k] for k in self.param_names], dtype=torch.float32
        )  # (P, 2)
        self.param_dim = len(self.param_names)
        self.max_pulses = max_pulses
        self.d_model = d_model

        # ------------------------------------------------------------------
        # Embedding layers
        # ------------------------------------------------------------------
        # W_delta: maps (delta_start, delta_end) -> (1, 8) tensor
        self.delta_emb = nn.Linear(2, 8)
        
        # Project the concatenated (10, 8) sequence to d_model
        # Input: (B, 10*n, 8) -> Output: (B, 10*n, d_model)
        self.input_proj = nn.Linear(8, d_model)

        # ------------------------------------------------------------------
        # Transformer Encoder
        # ------------------------------------------------------------------
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        if n_layers is None:
            n_layers = 4 * max_pulses

        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Output linear head – maps encoder hidden → pulse parameters (normalised)
        self.head = nn.Linear(d_model, self.max_pulses * self.param_dim)


    def forward(
        self, 
        rotation_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            rotation_vector: shape (B, n, 6) 
                Each element is (delta_start, delta_end, n_x, n_y, n_z, theta)
        
        Returns:
            pulses: shape (B, max_pulses, param_dim)
                Pulse parameters in physical units
        """
        B = rotation_vector.shape[0]
        n = rotation_vector.shape[1]

        # Extract delta range: (B, n, 2)
        delta_range = rotation_vector[..., :2]  # (B, n, 2)
        
        # Extract rotation parameters: (B, n, 4) where last dim is (n_x, n_y, n_z, theta)
        rotation_params = rotation_vector[..., 2:]  # (B, n, 4)

        # ------------------------------------------------------------------
        # 1. Compute delta embeddings: (B, n, 1, 8)
        # ------------------------------------------------------------------
        delta_emb = self.delta_emb(delta_range)  # (B, n, 8)
        delta_emb = delta_emb.unsqueeze(2)  # (B, n, 1, 8)

        # ------------------------------------------------------------------
        # 2. Compute SCORE embeddings: (B, n, 9, 8)
        # ------------------------------------------------------------------
        # YXY decomposition: (B, n, 3)
        euler_angles = UnitaryControlTransformer.euler_yxy_from_rotation_vector(
            rotation_params
        )  # (B, n, 3)

        
        # Get SCORE sequence: (B, n, 9, 2, 2) complex unitaries
        score_sequence = UnitaryControlTransformer.score_sequence_from_yxy(
            euler_angles
        )  # (B, n, 9, 2, 2)


        # Flatten to real vectors: (B, n, 9, 8)
        # Each 2x2 complex matrix -> 8 real numbers (4 complex = 8 reals)
        score_flat = UnitaryControlTransformer._to_real_vector(
            score_sequence
        )  # (B, n, 9, 8)


        # ------------------------------------------------------------------
        # 3. Concatenate delta and SCORE: (B, n, 10, 8)
        # ------------------------------------------------------------------
        # delta_emb: (B, n, 1, 8)
        # score_flat: (B, n, 9, 8)
        # Concatenate along the sequence dimension

        input_sequence = torch.cat([delta_emb, score_flat], dim=2)  # (B, n, 10, 8)

        # ------------------------------------------------------------------
        # 4. Reshape to (B, 10*n, 8) for processing
        # ------------------------------------------------------------------
        input_sequence = input_sequence.reshape(B, 10 * n, 8).to(torch.float32)  # (B, 10*n, 8)

        # ------------------------------------------------------------------
        # 5. Project to d_model: (B, 10*n, d_model)
        # ------------------------------------------------------------------
        emb = self.input_proj(input_sequence)  # (B, 10*n, d_model)

        # ------------------------------------------------------------------
        # 6. Add positional encoding
        # ------------------------------------------------------------------
        L = 10 * n  # Total sequence length
        pos_emb = UnitaryControlTransformer.sinusoidal_positional_encoding(
            L, self.d_model, device=emb.device
        )  # (L, d_model)
        
        emb = emb + pos_emb.unsqueeze(0)  # (B, L, d_model)

        # ------------------------------------------------------------------
        # 7. Transformer Encoder with self-attention across 10*n tokens
        # ------------------------------------------------------------------
        encoded = self.encoder(emb)  # (B, L, d_model)
        
        # ------------------------------------------------------------------
        # 8. Output head - use last token or pooled representation
        # ------------------------------------------------------------------
        # Option 1: Use the last token
        # output = self.head(encoded[:, -1, :])  # (B, max_pulses * param_dim)
        
        # Option 2: Use mean pooling (alternative)
        output = self.head(encoded.mean(dim=1))  # (B, max_pulses * param_dim)

        # ------------------------------------------------------------------
        # 9. Reshape to (B, max_pulses, param_dim)
        # ------------------------------------------------------------------
        pulses_norm = output.view(B, self.max_pulses, self.param_dim)

        # ------------------------------------------------------------------
        # 10. Map normalized outputs to physical parameter range
        # ------------------------------------------------------------------
        pulses_unit = pulses_norm.sigmoid()  # Map to [0, 1]
        low = self.param_ranges[:, 0].to(pulses_unit.device)
        high = self.param_ranges[:, 1].to(pulses_unit.device)

        pulses = low + (high - low) * pulses_unit  # (B, max_pulses, param_dim)

        return pulses  # (B, max_pulses, param_dim)



    @staticmethod
    def euler_yxy_from_rotation_vector(rotation_vector: torch.Tensor,
                                eps: float = 1e-12) -> torch.Tensor:
        """
        Vectorised Y-X-Y Euler decomposition.
        Args
        ----
            rotation_vector : (B,4) tensor (n_x, n_y, n_z, θ)
        Returns
        -------
            (B,3) tensor (α, β, γ) such that
                exp(-i θ/2 n·σ) = R_y(α) · R_x(β) · R_y(γ)
        """
        n, theta = rotation_vector[..., :3], rotation_vector[..., 3]
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(eps)       # normalise axis

        s, c = torch.sin(theta / 2), torch.cos(theta / 2)          # sin, cos θ/2
        w, x, y, z = c, n[..., 0] * s, n[..., 1] * s, n[..., 2] * s

        # ----- regular branch -------------------------------------------------- #
        beta = torch.acos((1.0 - 2.0 * (x**2 + z**2)).clamp(-1.0 + eps, 1.0 - eps))
        sin_beta = beta.sin()

        alpha_reg = torch.atan2(x * y + z * w, w * x - y * z)
        gamma_reg = torch.atan2(x * y - z * w, w * x + y * z)

        # ----- gimbal-lock handling ------------------------------------------- #
        tol = 1e-6
        mask_reg  = sin_beta.abs() > tol          # "normal" points
        mask_beta0 = ~mask_reg & (beta < 0.5)     # β ≈ 0         → Y-only
        mask_betapi = ~mask_reg & ~mask_beta0     # β ≈ π         → X/Z

        alpha = torch.where(mask_reg, alpha_reg, torch.zeros_like(alpha_reg))
        gamma = torch.where(mask_reg, gamma_reg, torch.zeros_like(gamma_reg))

        # β ≈ 0  (rotation about Y)
        # When sin(β/2) ≈ 0, we have x ≈ z ≈ 0, so rotation is about Y
        # α + γ = 2*atan2(y, w), we set γ=0 conventionally
        alpha = torch.where(mask_beta0, 2.0 * torch.atan2(y, w), alpha)
        gamma = torch.where(mask_beta0, torch.zeros_like(gamma), gamma)

        # β ≈ π  (rotation about X or Z)
        # When sin(β/2) ≈ 1, we have w ≈ y ≈ 0
        # α - γ = 2*atan2(z, x), we set α=0 conventionally  
        alpha = torch.where(mask_betapi, torch.zeros_like(alpha), alpha)
        gamma = torch.where(mask_betapi, 2.0 * torch.atan2(z, x), gamma)

        return torch.stack((alpha, beta, gamma), dim=-1)



    # ------------------------------------------------------------------ #
    # Low-level utilities                                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def unit_vec(phi: float,
                 dtype: torch.dtype = torch.float64,
                 device=None) -> torch.Tensor:
        """Unit vector (cos φ, sin φ, 0) lying in the x–y plane."""
        return torch.tensor([math.cos(phi), torch.sin(phi), 0.0],
                            dtype=dtype, device=device)

    @staticmethod
    def rotation_unitary(n: torch.Tensor,
                         angle: float,
                         dtype: torch.dtype = torch.complex128) -> torch.Tensor:
        """
        Returns the unitary for rotation about axis n by angle.
        Supports batched n and angle.
        n: (..., 3)
        angle: (...,)
        Returns: (..., 2, 2)
        """
        # Ensure n and angle are tensors
        n = torch.as_tensor(n)
        angle = torch.as_tensor(angle, device=n.device, dtype=n.dtype)
        # Broadcast
        x, y, z = n[..., 0], n[..., 1], n[..., 2]
        c = torch.cos(angle / 2)
        s = -1j * torch.sin(angle / 2)
        # Build the matrix using stack and cat for batch support
        row0 = torch.stack([c + s * z, s * (x - 1j * y)], dim=-1)
        row1 = torch.stack([s * (x + 1j * y), c - s * z], dim=-1)
        U = torch.stack([row0, row1], dim=-2)
        return U.to(dtype)
    
    # ------------------------------------------------------------------ #
    # SCORE composite pulse                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def get_score_emb_unitary(phi: float,
                              angle: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            score_tensor  – (3,2,2) composite implementing the SCORE pulse
            target_unitary – (2,2) ideal rotation R_{n=unit_vec(φ)}(θ)
        """
        theta = torch.pi - angle - torch.asin(0.5 * torch.sin(angle / 2))

        pulses: List[torch.Tensor] = [
            UnitaryControlTransformer.rotation_unitary(
                UnitaryControlTransformer.unit_vec(phi + torch.pi), theta),
            UnitaryControlTransformer.rotation_unitary(
                UnitaryControlTransformer.unit_vec(phi), phi + 2 * theta),
            UnitaryControlTransformer.rotation_unitary(
                UnitaryControlTransformer.unit_vec(phi + torch.pi), theta)
        ]

        score_tensor = torch.stack(pulses)                       # (3,2,2)
        target_unitary = UnitaryControlTransformer.rotation_unitary(
            UnitaryControlTransformer.unit_vec(phi), angle)

        return score_tensor, target_unitary


    # ------------------------------------------------------------------ #
    # Build (B,9,2,2) tensor from Y-X-Y angles                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def score_sequence_from_yxy(euler_angles: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        euler_angles : (B, n, 3) real tensor of Y-X-Y Euler triples (α, β, γ)

        Returns
        -------
        (B, n, 9, 2, 2) complex tensor whose 9 unitaries are
        [ SCORE(0, α) • SCORE(π/2, β) • SCORE(0, γ) ]  (three pulses each)
        """
        assert euler_angles.shape[-1] == 3, "expected (..., 3) Euler input"

        # Get the shape for proper reshaping
        original_shape = euler_angles.shape[:-1]  # (B, n)
        
        # Reshape to (B*n, 3) for batch processing
        euler_flat = euler_angles.reshape(-1, 3)  # (B*n, 3)
        
        # ---------- Inner helper: one sample -> nine unitaries ---------- #
        def _single_sequence(angles: torch.Tensor) -> torch.Tensor:
            """
            Args:
                angles: (3,) tensor of (α, β, γ)
            Returns:
                (9, 2, 2) complex tensor
            """
            alpha, beta, gamma = angles.unbind()
            phis = torch.tensor([0.0, math.pi / 2, 0.0], 
                            dtype=angles.dtype, 
                            device=angles.device)
            thetas = torch.stack([alpha, beta, gamma])

            blocks = [
                UnitaryControlTransformer.get_score_emb_unitary(phi, theta)[0]
                for phi, theta in zip(phis, thetas)
            ]
            return torch.cat(blocks, dim=0)  # (9, 2, 2)

        # vmap lifts the single-sample function to operate on the flattened batch
        score_flat = vmap(_single_sequence)(euler_flat)  # (B*n, 9, 2, 2)
        
        # Reshape back to (B, n, 9, 2, 2)
        score_output = score_flat.reshape(*original_shape, 9, 2, 2)
        
        return score_output.to(euler_angles.device)



    ###############################################################################
    # Utility helpers – flattening unitaries and fidelity
    ###############################################################################
    @staticmethod
    def _to_real_vector(U: torch.Tensor) -> torch.Tensor:
        """Flatten a complex matrix into a real‑valued vector with alternating real and imag components (…, 2*d*d)."""
        real = U.real.reshape(*U.shape[:-2], -1)  # shape: (..., d*d)
        imag = U.imag.reshape(*U.shape[:-2], -1)  # shape: (..., d*d)

        stacked = torch.stack((real, imag), dim=-1)  # shape: (..., d*d, 2)
        interleaved = stacked.reshape(*U.shape[:-2], -1)  # shape: (..., 2*d*d)

        return interleaved

    @staticmethod
    def fidelity(U_out: torch.Tensor, U_target: torch.Tensor) -> torch.Tensor:
        """
        Entanglement fidelity F = |Tr(U_out† U_target)|² / d²
        Works for any batch size B and dimension d.
        """
        d = U_out.shape[-1]

        # Trace of U_out† U_target – no data movement thanks to index relabelling.
        inner = torch.einsum("bji,bij->b", U_out.conj(), U_target)  # shape [B]

        return (inner.conj() * inner / d ** 2).real  # shape [B] or scalar

    ###############################################################################
    # Positional Embedding
    ###############################################################################

    @staticmethod
    def sinusoidal_positional_encoding(length: int, d_model: int, device: torch.device) -> torch.Tensor:
        """
        Generate sinusoidal positional encoding.

        Args:
            length: sequence length (e.g., max_pulses)
            d_model: embedding dimension
            device: torch device

        Returns:
            Tensor of shape (length, d_model)
        """
        position = torch.arange(length, dtype=torch.float, device=device).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))  # (d_model/2)

        pe = torch.zeros(length, d_model, device=device)  # (L, D)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe  # (L, D)

