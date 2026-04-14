"""Unit tests for depth back-projection and RLE encoding."""

import numpy as np
import pytest

from go2_open_vocab_detector.depth_backproject import (
    CameraIntrinsics,
    backproject_pixel,
    mask_rle_decode,
    mask_rle_encode,
    masked_depth_median,
    object_centroid_3d,
)


def _simple_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)


def test_backproject_centre_is_on_optical_axis():
    intr = _simple_intrinsics()
    p = backproject_pixel(intr.cx, intr.cy, depth_m=1.5, intr=intr)
    assert np.allclose(p, np.array([0.0, 0.0, 1.5]))


def test_backproject_offcenter_matches_pinhole():
    intr = _simple_intrinsics()
    # 100 px right of centre at 2 m depth → x = 100 * 2 / 600
    p = backproject_pixel(intr.cx + 100.0, intr.cy, depth_m=2.0, intr=intr)
    assert p[2] == pytest.approx(2.0)
    assert p[0] == pytest.approx(100.0 * 2.0 / 600.0)


def test_masked_depth_median_rejects_zeros_and_out_of_range():
    mask = np.ones((10, 10), dtype=bool)
    depth = np.zeros((10, 10), dtype=np.uint16)
    depth[0:5, :] = 1500  # 1.5 m
    depth[5:8, :] = 0     # invalid
    depth[8:, :] = 20000  # 20 m, out of range
    m = masked_depth_median(depth, mask, min_depth_m=0.2, max_depth_m=8.0)
    assert m == pytest.approx(1.5)


def test_masked_depth_median_all_invalid_returns_nan():
    mask = np.ones((10, 10), dtype=bool)
    depth = np.zeros((10, 10), dtype=np.uint16)
    assert np.isnan(masked_depth_median(depth, mask))


def test_object_centroid_3d_basic():
    intr = _simple_intrinsics()
    depth = np.full((intr.height, intr.width), 2000, dtype=np.uint16)  # 2 m everywhere
    mask = np.zeros_like(depth, dtype=bool)
    # 100x100 mask centred on image centre
    y0, y1 = intr.height // 2 - 50, intr.height // 2 + 50
    x0, x1 = intr.width // 2 - 50, intr.width // 2 + 50
    mask[y0:y1, x0:x1] = True

    centroid, depth_m, dims = object_centroid_3d(mask, depth, intr)
    assert depth_m == pytest.approx(2.0)
    # Centroid should be at image centre → optical axis → x≈0, y≈0.
    # Half-pixel residual at 2 m / fx 600 ≈ 1.67 mm, so allow ≤3 mm.
    assert abs(centroid[0]) < 3e-3
    assert abs(centroid[1]) < 3e-3
    assert centroid[2] == pytest.approx(2.0)
    # 100 px @ 2 m @ 600 fx ≈ 0.333 m
    assert dims[0] == pytest.approx(100 * 2.0 / 600.0, rel=1e-3)


def test_rle_roundtrip_random_masks():
    rng = np.random.default_rng(42)
    for _ in range(10):
        h = int(rng.integers(5, 40))
        w = int(rng.integers(5, 40))
        mask = rng.integers(0, 2, size=(h, w)).astype(bool)
        rle = mask_rle_encode(mask)
        recon = mask_rle_decode(rle, h, w)
        assert np.array_equal(mask, recon)


def test_rle_all_zeros_and_all_ones():
    mask_zeros = np.zeros((5, 7), dtype=bool)
    mask_ones = np.ones((5, 7), dtype=bool)
    for m in (mask_zeros, mask_ones):
        rle = mask_rle_encode(m)
        assert np.array_equal(mask_rle_decode(rle, m.shape[0], m.shape[1]), m)
