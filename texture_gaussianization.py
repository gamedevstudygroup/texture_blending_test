"""Gaussian texture mappings and joint-channel accuracy measurements.

The functions live outside the marimo notebook so they can be tested without
executing UI cells or the intentionally retained handwritten NetworkSimplex.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri

GAUSSIAN_MEAN = 0.5
GAUSSIAN_STD = 1.0 / 6.0
_AXIS_EPSILON = 1.0e-12
_COMPRESSION_SCALE_FLOOR = 1.0e-6


@dataclass(frozen=True)
class ColorSpace:
    """Forward working-space data and the affine transform back to RGB."""

    decorrelated: bool
    mean: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    minimum: NDArray[np.float64]
    span: NDArray[np.float64]
    origin: NDArray[np.float64]
    vectors: NDArray[np.float64]

    def to_rgb(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return normalized working-space values to the source RGB space."""
        array = np.asarray(values, dtype=np.float64)
        if not self.decorrelated:
            return array.copy()
        return self.origin + array @ self.vectors.T


@dataclass(frozen=True)
class SeparableResult:
    """GPU Zen 2 mapping plus reversible colorspace/storage metadata."""

    texture: NDArray[np.float64]
    canonical_texture: NDArray[np.float64]
    working_source: NDArray[np.float64]
    colorspace: ColorSpace
    compression_scale: NDArray[np.float64]

    def undo_compression_correction(self) -> NDArray[np.float64]:
        """Recover canonical Gaussian values from the storage-scaled texture."""
        return GAUSSIAN_MEAN + (
            (self.texture - GAUSSIAN_MEAN) / self.compression_scale
        )


@dataclass(frozen=True)
class DistributionMetrics:
    """3D reference error and dependence retained between output channels."""

    reference_js: float
    dependence_js: float
    reference_dependence_js: float
    excess_dependence_js: float


def as_float_rgb(image: ArrayLike) -> NDArray[np.float64]:
    """Validate an RGB image and normalize integer pixels to the unit cube."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.size == 0:
        raise ValueError("image must be a non-empty height x width x 3 array")
    if not np.all(np.isfinite(array)):
        raise ValueError("image must contain only finite values")
    result = array.astype(np.float64)
    if np.issubdtype(array.dtype, np.integer):
        result /= np.iinfo(array.dtype).max
    elif result.min() < 0.0 or result.max() > 1.0:
        raise ValueError("floating-point images must be in the [0, 1] range")
    return result


def gaussian_reference(
    shape: int | tuple[int, int] | np.ndarray,
    *,
    n_channels: int = 3,
    seed: int = 1,
    mean: float = GAUSSIAN_MEAN,
    std: float = GAUSSIAN_STD,
) -> NDArray[np.float64]:
    """Create independently coupled, jitter-stratified 3D Gaussian samples."""
    count = np.asarray(shape).prod()
    if count <= 0 or std <= 0.0:
        raise ValueError("count and standard deviation must be positive")

    rng = np.random.default_rng(seed)
    channels = []
    for _ in range(n_channels):
        # Jittered strata closely sample the Gaussian CDF without the exact
        # mirror symmetries of midpoint quantiles, then an independent
        # permutation removes any rank coupling between channels.
        quantiles = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
        values = mean + std * ndtri(quantiles)
        channels.append(rng.permutation(values))
    out = np.column_stack(channels)
    return out.reshape(*np.atleast_1d(shape), n_channels)

def _oriented_eigenvectors(covariance: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a deterministic, right-handed PCA basis ordered by variance."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    # Eigenvector signs are arbitrary; fixing the largest component prevents
    # previews and metadata from flipping across otherwise identical runs.
    for axis in range(3):
        pivot = int(np.argmax(np.abs(eigenvectors[:, axis])))
        if eigenvectors[pivot, axis] < 0.0:
            eigenvectors[:, axis] *= -1.0
    if np.linalg.det(eigenvectors) < 0.0:
        eigenvectors[:, -1] *= -1.0
    return eigenvectors


def decorrelated_colorspace(
    image: ArrayLike, *, enabled: bool = True
) -> tuple[NDArray[np.float64], ColorSpace]:
    """Map RGB pixels to the paper's PCA-aligned unit bounding box."""
    rgb = as_float_rgb(image)
    pixels = rgb.reshape(-1, 3)
    if not enabled:
        minimum = pixels.min(axis=0)
        span = pixels.max(axis=0) - minimum
        metadata = ColorSpace(
            decorrelated=False,
            mean=np.zeros(3),
            eigenvectors=np.eye(3),
            minimum=minimum,
            span=span,
            origin=np.zeros(3),
            vectors=np.eye(3),
        )
        return pixels.copy(), metadata

    mean = pixels.mean(axis=0)
    centered = pixels - mean
    covariance = centered.T @ centered / max(len(pixels) - 1, 1)
    eigenvectors = _oriented_eigenvectors(covariance)
    projected = centered @ eigenvectors
    minimum = projected.min(axis=0)
    span = projected.max(axis=0) - minimum
    safe_span = np.where(span > _AXIS_EPSILON, span, 1.0)
    working = (projected - minimum) / safe_span
    # A collapsed principal axis has no meaningful coordinate. Centering it is
    # stable and reconstructs exactly because its stored basis vector is zero.
    working[:, span <= _AXIS_EPSILON] = 0.5

    origin = mean + minimum @ eigenvectors.T
    vectors = eigenvectors * span[np.newaxis, :]
    metadata = ColorSpace(
        decorrelated=True,
        mean=mean,
        eigenvectors=eigenvectors,
        minimum=minimum,
        span=span,
        origin=origin,
        vectors=vectors,
    )
    return working, metadata


def separable_gaussianize(
    image: ArrayLike,
    *,
    reference: ArrayLike | None = None,
    seed: int = 1,
    decorrelate: bool = True,
    compression_correction: bool = False,
) -> SeparableResult:
    """Apply the GPU Zen 2 per-channel rank transform in a PCA color space."""
    rgb = as_float_rgb(image)
    working, colorspace = decorrelated_colorspace(rgb, enabled=decorrelate)
    target = (
        gaussian_reference(len(working), seed=seed)
        if reference is None
        else np.asarray(reference, dtype=np.float64)
    )
    if target.shape != working.shape or not np.all(np.isfinite(target)):
        raise ValueError("reference must have the same pixel count and three channels")

    # Listing 2.8 in GPU Zen 2 maps rank i to the midpoint quantile
    # (i + 0.5) / N. This is intentionally distinct from the independently
    # jittered 3D reference used by the joint-distribution accuracy metric.
    quantiles = (np.arange(len(working), dtype=np.float64) + 0.5) / len(working)
    gaussian_ranks = GAUSSIAN_MEAN + GAUSSIAN_STD * ndtri(quantiles)
    canonical = np.empty_like(working)
    for channel in range(3):
        # Stable sorting makes equal-valued source pixels deterministic. Each
        # channel receives exactly the reference marginal, but its cross-channel
        # coupling continues to come from the source texture.
        order = np.argsort(working[:, channel], kind="stable")
        canonical[order, channel] = gaussian_ranks

    compression_scale = np.ones(3, dtype=np.float64)
    if compression_correction:
        widest = float(np.max(colorspace.span))
        if widest > _AXIS_EPSILON:
            compression_scale = np.maximum(
                colorspace.span / widest, _COMPRESSION_SCALE_FLOOR
            )
    encoded = GAUSSIAN_MEAN + (
        canonical - GAUSSIAN_MEAN
    ) * compression_scale
    return SeparableResult(
        texture=encoded.reshape(rgb.shape),
        canonical_texture=canonical.reshape(rgb.shape),
        working_source=working.reshape(rgb.shape),
        colorspace=colorspace,
        compression_scale=compression_scale,
    )


def _quantile_histogram(samples: ArrayLike, bins: int) -> NDArray[np.float64]:
    """Histogram the Gaussian copula so marginal scale cannot hide dependence."""
    values = np.asarray(samples, dtype=np.float64).reshape(-1, 3)
    if bins < 2:
        raise ValueError("bins must be at least two")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must contain only finite values")
    quantiles = ndtr((values - GAUSSIAN_MEAN) / GAUSSIAN_STD)
    histogram, _ = np.histogramdd(
        quantiles, bins=bins, range=((0.0, 1.0),) * 3
    )
    histogram /= histogram.sum()
    return histogram


def _js_divergence(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> float:
    """Return base-2 Jensen-Shannon divergence, bounded between zero and one."""
    p = np.asarray(left, dtype=np.float64).ravel()
    q = np.asarray(right, dtype=np.float64).ravel()
    midpoint = 0.5 * (p + q)
    p_mask = p > 0.0
    q_mask = q > 0.0
    divergence = 0.5 * np.sum(p[p_mask] * np.log2(p[p_mask] / midpoint[p_mask]))
    divergence += 0.5 * np.sum(q[q_mask] * np.log2(q[q_mask] / midpoint[q_mask]))
    return float(divergence)


def _dependence_js(histogram: NDArray[np.float64]) -> float:
    """Compare a joint histogram with the product of its own marginals."""
    x = histogram.sum(axis=(1, 2))
    y = histogram.sum(axis=(0, 2))
    z = histogram.sum(axis=(0, 1))
    independent = x[:, None, None] * y[None, :, None] * z[None, None, :]
    return _js_divergence(histogram, independent)


def distribution_metrics(
    samples: ArrayLike, reference: ArrayLike, *, bins: int = 8
) -> DistributionMetrics:
    """Score 3D Gaussian closeness and nonlinear cross-channel dependence."""
    sample_histogram = _quantile_histogram(samples, bins)
    reference_histogram = _quantile_histogram(reference, bins)
    dependence = _dependence_js(sample_histogram)
    reference_dependence = _dependence_js(reference_histogram)
    return DistributionMetrics(
        reference_js=_js_divergence(sample_histogram, reference_histogram),
        dependence_js=dependence,
        reference_dependence_js=reference_dependence,
        excess_dependence_js=dependence - reference_dependence,
    )


__all__ = [
    "GAUSSIAN_MEAN",
    "GAUSSIAN_STD",
    "ColorSpace",
    "DistributionMetrics",
    "SeparableResult",
    "as_float_rgb",
    "decorrelated_colorspace",
    "distribution_metrics",
    "gaussian_reference",
    "separable_gaussianize",
]
