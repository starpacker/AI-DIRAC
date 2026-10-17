"""
Unified observation data model — Phase 1.

Defines a protocol that allows the system to handle:
  - TrajectoryData: deterministic time series (classical mechanics)
  - EventEnsembleData: repeated random-event collections (wave-particle)
  - FieldGridData: spatial field data (GR / field theory)

Every observation carries:
  - raw readings
  - sampling time or event order
  - measurement context (apparatus, parameters)
  - object/event identity
  - control and intervention records
  - noise model and uncertainty
  - apparatus family and parameter range
  - allowed derived quantities with full provenance
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from .provenance import ChannelOrigin, ChannelProvenance, ProvenanceTracker


# ─── Measurement context ─────────────────────────────────────────────────

@dataclass
class MeasurementContext:
    """Context in which a measurement was taken."""
    apparatus_name: str = ""
    parameter_values: Dict[str, float] = field(default_factory=dict)
    intervention_spec: Optional[str] = None  # e.g., "change mass to 2.0"
    noise_model: str = "gaussian"
    noise_std: float = 0.01
    sampling_rate: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "apparatus_name": self.apparatus_name,
            "parameter_values": self.parameter_values,
            "intervention_spec": self.intervention_spec,
            "noise_model": self.noise_model,
            "noise_std": self.noise_std,
            "sampling_rate": self.sampling_rate,
            "extra": self.extra,
        }


# ─── Abstract observation base ───────────────────────────────────────────

class Observation(ABC):
    """Base class for all observation types."""

    def __init__(self):
        self._provenance: Dict[str, ChannelProvenance] = {}

    @property
    @abstractmethod
    def observation_type(self) -> str:
        """Return 'trajectory', 'event_ensemble', or 'field_grid'."""
        ...

    @abstractmethod
    def get_channel(self, channel_name: str) -> Optional[np.ndarray]:
        """Get raw readings for a channel."""
        ...

    @abstractmethod
    def get_channels(self) -> List[str]:
        """List all available channel names."""
        ...

    def set_provenance(self, channel_name: str, prov: ChannelProvenance) -> None:
        self._provenance[channel_name] = prov

    def get_provenance(self, channel_name: str) -> Optional[ChannelProvenance]:
        return self._provenance.get(channel_name)

    def get_all_provenance(self) -> Dict[str, ChannelProvenance]:
        return dict(self._provenance)


# ─── Trajectory data ─────────────────────────────────────────────────────

@dataclass
class TrajectoryData(Observation):
    """
    Classical time-series trajectory from an ODE simulator or experiment.

    Each channel is a 1-D array indexed by time step.
    """
    channels: Dict[str, np.ndarray] = field(default_factory=dict)
    time: Optional[np.ndarray] = None
    context: MeasurementContext = field(default_factory=MeasurementContext)

    def __post_init__(self):
        super().__init__()
        if self.time is None and "t" in self.channels:
            self.time = self.channels["t"]

    @property
    def observation_type(self) -> str:
        return "trajectory"

    def get_channel(self, channel_name: str) -> Optional[np.ndarray]:
        return self.channels.get(channel_name)

    def get_channels(self) -> List[str]:
        return list(self.channels.keys())

    def num_points(self) -> int:
        if self.time is not None:
            return len(self.time)
        # Use first channel length
        for arr in self.channels.values():
            return len(arr)
        return 0

    def slice_time(self, t_start: float, t_end: float) -> "TrajectoryData":
        """Return a time-sliced copy."""
        if self.time is None:
            return self
        mask = (self.time >= t_start) & (self.time <= t_end)
        new_channels = {
            k: v[mask] for k, v in self.channels.items()
        }
        return TrajectoryData(
            channels=new_channels,
            time=self.time[mask],
            context=self.context,
        )

    def held_out_split(self, fraction: float = 0.2) -> Tuple["TrajectoryData", "TrajectoryData"]:
        """
        Split into training and held-out sets based on time segments.
        Returns (train, held_out).
        """
        n = self.num_points()
        split_idx = int(n * (1 - fraction))
        train = TrajectoryData(
            channels={k: v[:split_idx] for k, v in self.channels.items()},
            time=self.time[:split_idx] if self.time is not None else None,
            context=self.context,
        )
        held_out = TrajectoryData(
            channels={k: v[split_idx:] for k, v in self.channels.items()},
            time=self.time[split_idx:] if self.time is not None else None,
            context=self.context,
        )
        return train, held_out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "trajectory",
            "channels": {k: v.tolist() for k, v in self.channels.items()},
            "context": self.context.to_dict(),
            "provenance": {
                k: prov.to_dict() for k, prov in self._provenance.items()
            },
        }


# ─── Event ensemble data ─────────────────────────────────────────────────

@dataclass
class EventEnsembleData(Observation):
    """
    Collection of repeated event-based measurements.

    Each channel is 2-D: (event_index, measurement).
    Suitable for quantum/statistical experiments.
    """
    channels: Dict[str, np.ndarray] = field(default_factory=dict)
    event_ids: Optional[np.ndarray] = None
    event_times: Optional[np.ndarray] = None
    context: MeasurementContext = field(default_factory=MeasurementContext)

    def __post_init__(self):
        super().__init__()
        if self.event_ids is None:
            first_arr = next(iter(self.channels.values()), None)
            if first_arr is not None:
                self.event_ids = np.arange(len(first_arr))

    @property
    def observation_type(self) -> str:
        return "event_ensemble"

    def get_channel(self, channel_name: str) -> Optional[np.ndarray]:
        return self.channels.get(channel_name)

    def get_channels(self) -> List[str]:
        return list(self.channels.keys())

    def num_events(self) -> int:
        if self.event_ids is not None:
            return len(self.event_ids)
        return 0

    def conditional_distribution(
        self, channel_name: str, condition_channel: str, condition_value: float,
        tolerance: float = 0.1
    ) -> Optional[np.ndarray]:
        """Return distribution of channel_name conditioned on condition_channel ≈ condition_value."""
        data = self.channels.get(channel_name)
        cond = self.channels.get(condition_channel)
        if data is None or cond is None:
            return None
        mask = np.abs(cond - condition_value) < tolerance
        return data[mask]

    def bootstrap_sample(self, n_samples: int = 100) -> "EventEnsembleData":
        """Return a bootstrap resample."""
        n = self.num_events()
        indices = np.random.choice(n, n_samples, replace=True)
        new_channels = {
            k: v[indices] for k, v in self.channels.items()
        }
        return EventEnsembleData(
            channels=new_channels,
            event_ids=np.arange(n_samples),
            context=self.context,
        )


# ─── Field grid data ─────────────────────────────────────────────────────

@dataclass
class FieldGridData(Observation):
    """
    Spatial field data on a regular grid.

    Each channel is an N-D array indexed by spatial coordinates.
    Suitable for general relativity / continuum mechanics.
    """
    channels: Dict[str, np.ndarray] = field(default_factory=dict)
    grid_axes: Dict[str, np.ndarray] = field(default_factory=dict)
    context: MeasurementContext = field(default_factory=MeasurementContext)

    def __post_init__(self):
        super().__init__()

    @property
    def observation_type(self) -> str:
        return "field_grid"

    def get_channel(self, channel_name: str) -> Optional[np.ndarray]:
        return self.channels.get(channel_name)

    def get_channels(self) -> List[str]:
        return list(self.channels.keys())

    def num_points(self) -> int:
        """Total number of grid points."""
        n = 1
        for axis in self.grid_axes.values():
            n *= len(axis)
        return n

    def spatial_held_out_split(
        self, fraction: float = 0.2, axis: str = "x"
    ) -> Tuple["FieldGridData", "FieldGridData"]:
        """
        Split into training and held-out sets along a spatial axis.
        """
        if axis not in self.grid_axes:
            return self, self
        axis_idx = list(self.grid_axes.keys()).index(axis)
        n = len(self.grid_axes[axis])
        split_idx = int(n * (1 - fraction))

        train_slices = [slice(None)] * len(self.grid_axes)
        test_slices = [slice(None)] * len(self.grid_axes)
        train_slices[axis_idx] = slice(0, split_idx)
        test_slices[axis_idx] = slice(split_idx, n)

        train = FieldGridData(
            channels={k: v[tuple(train_slices)] for k, v in self.channels.items()},
            grid_axes={
                k: v[:split_idx] if k == axis else v
                for k, v in self.grid_axes.items()
            },
            context=self.context,
        )
        held_out = FieldGridData(
            channels={k: v[tuple(test_slices)] for k, v in self.channels.items()},
            grid_axes={
                k: v[split_idx:] if k == axis else v
                for k, v in self.grid_axes.items()
            },
            context=self.context,
        )
        return train, held_out
