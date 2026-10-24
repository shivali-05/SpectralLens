"""Windowing utilities enforcing strict split containment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spectrallens.data.protocol import Protocol


@dataclass(frozen=True)
class Windows:
    """A collection of sliced windows and their coordinates.

    Attributes:
        data: Array of shape (n_windows, window_size), float64.
        indices: Array of shape (n_windows, 2), int64. Each row is [start, end)
            relative to the original 1D series.
        split_name: Name of the split ('fit', 'calibration', 'test', etc.) or empty.
    """

    data: np.ndarray
    indices: np.ndarray
    split_name: str = ""

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(
                f"Windows data must be 2D (n_windows, window_size), got shape {self.data.shape}"
            )
        if self.indices.ndim != 2 or self.indices.shape[1] != 2:
            raise ValueError(
                f"Windows indices must have shape (n_windows, 2), got shape {self.indices.shape}"
            )
        if len(self.data) != len(self.indices):
            raise ValueError(
                f"Mismatch between number of windows ({len(self.data)}) "
                f"and indices ({len(self.indices)})"
            )

    @property
    def n_windows(self) -> int:
        return len(self.data)

    @property
    def window_size(self) -> int:
        return self.data.shape[1] if self.n_windows > 0 else 0


def slice_windows(
    x: np.ndarray,
    window_size: int,
    step: int,
    start: int = 0,
    end: int | None = None,
    split_name: str = "",
) -> Windows:
    """Slice a 1D series into fixed-size windows strictly within [start, end).

    Guarantees that every extracted window [w_start, w_end) satisfies:
        start <= w_start < w_end <= end

    Args:
        x: 1D input array.
        window_size: Length of each window in samples.
        step: Stride between successive window start positions.
        start: Inclusive start sample index of the valid region.
        end: Exclusive end sample index of the valid region. If None, len(x) is used.
        split_name: Optional label for the split ('fit', 'calibration', 'test').

    Returns:
        Windows object holding the (n_windows, window_size) array and absolute [start, end) indices.
    """
    if x.ndim != 1:
        raise ValueError(f"Input series x must be 1D, got ndim={x.ndim}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")

    total_len = len(x)
    end = total_len if end is None else end

    if not (0 <= start <= end <= total_len):
        raise ValueError(f"Invalid range [{start}, {end}) for series of length {total_len}")

    span = end - start
    if span < window_size:
        return Windows(
            data=np.empty((0, window_size), dtype=x.dtype),
            indices=np.empty((0, 2), dtype=np.int64),
            split_name=split_name,
        )

    starts = np.arange(start, end - window_size + 1, step, dtype=np.int64)
    window_offsets = np.arange(window_size, dtype=np.int64)
    grid = starts[:, None] + window_offsets[None, :]
    data = x[grid]
    indices = np.column_stack((starts, starts + window_size))

    return Windows(data=data, indices=indices, split_name=split_name)


def extract_split_windows(
    x: np.ndarray,
    protocol: Protocol,
    window_size: int,
    step: int,
) -> dict[str, Windows]:
    """Extract windows strictly contained within each split of the protocol.

    Guarantees no window straddles split boundaries.

    Args:
        x: Full 1D time-series array of length protocol.series_length.
        protocol: Protocol instance specifying 'fit', 'calibration', and 'test' regions.
        window_size: Length of each window in samples.
        step: Stride between successive window start positions.

    Returns:
        Dict mapping split names ('fit', 'calibration', 'test') to their corresponding Windows.
    """
    if len(x) != protocol.series_length:
        raise ValueError(
            f"Series length {len(x)} does not match protocol series_length {protocol.series_length}"
        )

    return {
        name: slice_windows(
            x,
            window_size=window_size,
            step=step,
            start=span[0],
            end=span[1],
            split_name=name,
        )
        for name, span in protocol.splits().items()
    }
