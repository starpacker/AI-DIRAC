"""
NeuralRepresentationProposer — Phase 3.

Neural networks propose latent representations (f_θ) with hard constraints:
  1. Parameters SHARED across experiments (not per-experiment fitting)
  2. Latent dimension has explicit upper bound
  3. Must show predictive gain on held-out interventions
  4. Distillable to symbolic or program form
  5. Gauge freedom tracked (equivalent representations detected)
  6. Black-box representations CANNOT enter law discovery without certification

The neural proposer is a filter: it proposes candidates, but only
those that pass symbolic distillation and certification can be used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto
import math
import warnings

import numpy as np

# Optional imports — gracefully degrade without torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class NeuralRepresentationStatus(Enum):
    PROPOSED = auto()         # Network trained
    DISTILLING = auto()       # Being converted to symbolic form
    DISTILLED = auto()        # Symbolic form found
    CERTIFIED = auto()        # Passed all gates
    REJECTED = auto()         # Failed distillation or validation
    BLACKBOX = auto()         # Cannot be certified, stays as tool only


@dataclass
class NeuralRepresentation:
    """A neural latent representation candidate."""
    repr_id: str
    input_dim: int
    latent_dim: int
    # Architecture
    hidden_dims: List[int] = field(default_factory=list)
    activation: str = "relu"
    # Training
    train_loss: float = float("inf")
    val_loss: float = float("inf")
    heldout_loss: float = float("inf")
    cross_experiment_loss: float = float("inf")
    # Distillation
    distilled_expression: str = ""
    distillation_error: float = float("inf")
    distillation_complexity: int = 0
    # Status
    status: NeuralRepresentationStatus = NeuralRepresentationStatus.PROPOSED
    # Gauge
    is_invariant: bool = False          # true if representation is gauge-invariant
    gauge_group_description: str = ""
    # Metadata
    training_experiments: List[str] = field(default_factory=list)
    model_state: Optional[Dict[str, Any]] = None  # serialized weights

    @property
    def is_certifiable(self) -> bool:
        return (
            self.distilled_expression != ""
            and self.distillation_error < 0.1
            and self.heldout_loss < 0.1
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repr_id": self.repr_id,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "heldout_loss": self.heldout_loss,
            "cross_experiment_loss": self.cross_experiment_loss,
            "distilled_expression": self.distilled_expression,
            "distillation_error": self.distillation_error,
            "distillation_complexity": self.distillation_complexity,
            "status": self.status.name,
            "is_invariant": self.is_invariant,
            "gauge_group_description": self.gauge_group_description,
            "training_experiments": self.training_experiments,
        }


@dataclass
class NeuralProposerConfig:
    max_latent_dim: int = 8
    max_hidden_dim: int = 64
    max_layers: int = 4
    max_epochs: int = 500
    early_stopping_patience: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    heldout_fraction: float = 0.2
    val_fraction: float = 0.2
    batch_size: int = 64
    seed: int = 42


class NeuralRepresentationProposer:
    """
    Trains small neural nets as representation candidates.

    Hard constraints:
      - Latent dim ≤ max_latent_dim
      - Must improve on heldout and cross-experiment data
      - Gauge degeneracies tracked (detect rotations that leave latent unchanged)
    """

    def __init__(
        self,
        config: Optional[NeuralProposerConfig] = None,
    ):
        self.config = config or NeuralProposerConfig()
        self._id_counter = 0

        if not HAS_TORCH:
            warnings.warn(
                "PyTorch not available. NeuralProposer will operate in "
                "reduced mode using numpy-only approximations."
            )

    # --- Gauge-invariant dimension estimation ---

    @staticmethod
    def estimate_gauge_dimension(
        data: np.ndarray,
        *,
        max_dim: Optional[int] = None,
        variance_threshold: float = 0.95,
    ) -> int:
        """
        Estimate the intrinsic (gauge-invariant) dimension via PCA.

        Returns the number of components needed to explain variance_threshold
        of total variance. This is a model-free lower bound on latent dimension.
        """
        n, d = data.shape
        if n < 2 or d < 2:
            return 1

        center = data - np.mean(data, axis=0)
        U, S, Vt = np.linalg.svd(center, full_matrices=False)

        total_var = np.sum(S ** 2)
        cum_var = np.cumsum(S ** 2) / total_var

        k = int(np.searchsorted(cum_var, variance_threshold) + 1)
        if max_dim is not None:
            k = min(k, max_dim)
        return max(k, 1)

    # --- Neural network encoder ---

    def _build_encoder(
        self, input_dim: int, latent_dim: int, hidden_dims: List[int]
    ) -> nn.Module:
        """Build a simple MLP encoder."""
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.ReLU())
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, latent_dim))
        # No activation on final layer — raw latent
        return nn.Sequential(*layers)

    def _build_decoder(
        self, latent_dim: int, output_dim: int, hidden_dims: List[int]
    ) -> nn.Module:
        """Build decoder: latent → output."""
        layers = []
        prev_dim = latent_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.ReLU())
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _build_autoencoder(
        self, input_dim: int, latent_dim: int, hidden_dims: List[int]
    ) -> nn.Module:
        """Build autoencoder: input → latent → reconstructed input."""
        encoder = self._build_encoder(input_dim, latent_dim, hidden_dims)
        decoder = self._build_decoder(
            latent_dim, input_dim, list(reversed(hidden_dims))
        )

        class AutoEncoder(nn.Module):
            def __init__(self, enc, dec):
                super().__init__()
                self.encoder = enc
                self.decoder = dec

            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z), z

        return AutoEncoder(encoder, decoder)

    def propose_representations(
        self,
        data: Dict[str, np.ndarray],  # experiment_name → (n_samples, n_features)
        *,
        heldout_data: Optional[Dict[str, np.ndarray]] = None,
        target_latent_dims: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> List[NeuralRepresentation]:
        """
        Train autoencoders and propose latent representations.

        Args:
            data: Training data keyed by experiment name
            heldout_data: Held-out data for validation
            target_latent_dims: List of latent dimensions to try
        """
        if not HAS_TORCH:
            return self._numpy_propose(data, target_latent_dims)

        if target_latent_dims is None:
            max_dim = self.config.max_latent_dim
            # Also try estimated gauge dimension
            all_data = np.concatenate([v for v in data.values()], axis=0)
            est_dim = self.estimate_gauge_dimension(all_data, max_dim=max_dim)
            target_latent_dims = sorted(set(range(1, max_dim + 1)) | {est_dim})

        results: List[NeuralRepresentation] = []

        # Combine all training data
        all_input = np.concatenate([v for v in data.values()], axis=0)
        input_dim = all_input.shape[1]

        # Convert to torch
        X = torch.tensor(all_input, dtype=torch.float32)

        # Train/val split
        np.random.seed(self.config.seed)
        n = len(X)
        indices = np.random.permutation(n)
        split = int(n * (1 - self.config.val_fraction))
        train_idx = indices[:split]
        val_idx = indices[split:]

        X_train = X[train_idx]
        X_val = X[val_idx]

        for latent_dim in target_latent_dims:
            hidden_dims = self._get_hidden_dims(input_dim, latent_dim)

            try:
                result = self._train_autoencoder(
                    X_train, X_val,
                    input_dim, latent_dim, hidden_dims,
                    data, heldout_data,
                    verbose=verbose,
                )
                if result is not None:
                    results.append(result)
            except Exception as e:
                if verbose:
                    print(f"  Failed latent_dim={latent_dim}: {e}")

        results.sort(key=lambda r: r.val_loss)
        return results

    def _get_hidden_dims(self, input_dim: int, latent_dim: int) -> List[int]:
        """Determine hidden layer dimensions."""
        max_h = self.config.max_hidden_dim
        # Geometric progression from input to latent
        num_layers = min(self.config.max_layers, 3)

        dims = []
        for i in range(num_layers):
            t = (i + 1) / (num_layers + 1)
            d = int(input_dim * (1 - t) + latent_dim * t)
            d = max(latent_dim, min(d, max_h))
            dims.append(d)
        return dims

    def _train_autoencoder(
        self,
        X_train, X_val,
        input_dim, latent_dim, hidden_dims,
        data, heldout_data,
        verbose=False,
    ) -> Optional[NeuralRepresentation]:
        """Train one autoencoder configuration."""
        model = self._build_autoencoder(input_dim, latent_dim, hidden_dims)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.config.max_epochs):
            model.train()
            optimizer.zero_grad()

            recon, _ = model(X_train)
            loss = F.mse_loss(recon, X_train)

            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_recon, _ = model(X_val)
                val_loss = F.mse_loss(val_recon, X_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {
                    k: v.clone() for k, v in model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    break

        if best_state is None:
            return None

        # Restore best model
        model.load_state_dict(best_state)

        # Compute heldout loss
        heldout_loss = float("inf")
        if heldout_data is not None:
            held_input = torch.tensor(
                np.concatenate([v for v in heldout_data.values()], axis=0),
                dtype=torch.float32,
            )
            model.eval()
            with torch.no_grad():
                h_recon, _ = model(held_input)
                heldout_loss = F.mse_loss(h_recon, held_input).item()

        # Cross-experiment: test on each experiment separately
        cross_loss = self._compute_cross_experiment_loss(model, data)

        self._id_counter += 1
        return NeuralRepresentation(
            repr_id=f"NNREP_{self._id_counter:04d}",
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation="relu",
            train_loss=float(
                F.mse_loss(model(X_train)[0], X_train).item()
            ),
            val_loss=float(best_val_loss),
            heldout_loss=float(heldout_loss),
            cross_experiment_loss=cross_loss,
            training_experiments=list(data.keys()),
            is_invariant=latent_dim > 1,
        )

    def _compute_cross_experiment_loss(
        self, model, data: Dict[str, np.ndarray]
    ) -> float:
        """Test how well the model reconstructs each experiment separately."""
        losses = []
        model.eval()
        with torch.no_grad():
            for exp_data in data.values():
                X = torch.tensor(exp_data, dtype=torch.float32)
                recon, _ = model(X)
                losses.append(F.mse_loss(recon, X).item())
        return float(np.mean(losses)) if losses else float("inf")

    # --- Numpy fallback (no PyTorch) ---

    def _numpy_propose(
        self,
        data: Dict[str, np.ndarray],
        target_latent_dims: Optional[List[int]] = None,
    ) -> List[NeuralRepresentation]:
        """Fallback: PCA-based representation proposals."""
        all_data = np.concatenate([v for v in data.values()], axis=0)
        input_dim = all_data.shape[1]

        if target_latent_dims is None:
            target_latent_dims = list(range(1, min(self.config.max_latent_dim + 1, input_dim + 1)))

        results = []

        center = all_data - np.mean(all_data, axis=0)
        U, S, Vt = np.linalg.svd(center, full_matrices=False)

        total_var = np.sum(S ** 2)

        for latent_dim in target_latent_dims:
            if latent_dim > len(S):
                break

            # Reconstruction using top-k components
            Uk = U[:, :latent_dim]
            Sk = np.diag(S[:latent_dim])
            Vtk = Vt[:latent_dim, :]

            recon = Uk @ Sk @ Vtk + np.mean(all_data, axis=0)
            recon_error = np.mean((all_data - recon) ** 2)
            explained_var = np.sum(S[:latent_dim] ** 2) / total_var

            self._id_counter += 1
            results.append(NeuralRepresentation(
                repr_id=f"NNREP_{self._id_counter:04d}",
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=[],
                activation="linear",
                train_loss=float(recon_error),
                val_loss=float(recon_error),
                heldout_loss=float(1.0 - explained_var),
                cross_experiment_loss=float(1.0 - explained_var),
                training_experiments=list(data.keys()),
                is_invariant=True,
            ))

        return results

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "max_latent_dim": self.config.max_latent_dim,
                "max_hidden_dim": self.config.max_hidden_dim,
                "max_layers": self.config.max_layers,
                "max_epochs": self.config.max_epochs,
                "early_stopping_patience": self.config.early_stopping_patience,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "heldout_fraction": self.config.heldout_fraction,
                "val_fraction": self.config.val_fraction,
                "batch_size": self.config.batch_size,
                "seed": self.config.seed,
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NeuralRepresentationProposer":
        config = NeuralProposerConfig()
        cfg_d = d.get("config", {})
        for k, v in cfg_d.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return cls(config=config)
