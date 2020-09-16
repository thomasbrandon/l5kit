from typing import Callable

import numpy as np
import pytest
import transforms3d

from l5kit.geometry import SUBPIXEL_SHIFT, transform_point
from l5kit.geometry.transform import (
    transform_points_nb,
    transform_points_np,
    transform_points_subpixel_nb,
    transform_points_subpixel_np,
)


@pytest.mark.parametrize("transform_points", [transform_points_np, transform_points_nb])
def test_transform_points(transform_points: Callable) -> None:
    tf = np.asarray([[1.0, 0, 100], [0, 0.5, 50], [0, 0, 1]])

    points = np.array([[0, 10], [10, 0], [10, 10]])
    expected_point = np.array([[100, 55], [110, 50], [110, 55]])

    output_points = transform_points(points, tf)

    np.testing.assert_array_equal(output_points, expected_point)


def test_transform_single_point() -> None:
    tf = np.asarray([[1.0, 0, 100], [0, 0.5, 50], [0, 0, 1]])

    point = np.array([0, 10])
    expected_point = np.array([100, 55])

    output_point = transform_point(point, tf)

    np.testing.assert_array_equal(output_point, expected_point)


@pytest.mark.parametrize("transform_points", [transform_points_np, transform_points_nb])
def test_transform_points_revert_equivalence(transform_points: Callable) -> None:
    input_points = np.random.rand(10, 3)

    #  Generate some random transformation matrix
    tf = np.identity(4)
    tf[:3, :3] = transforms3d.euler.euler2mat(np.random.rand(), np.random.rand(), np.random.rand())
    tf[3, :3] = np.random.rand(3)

    output_points = transform_points(input_points, tf)

    tf_inv = np.linalg.inv(tf)

    input_points_recovered = transform_points(output_points, tf_inv)

    np.testing.assert_almost_equal(input_points_recovered, input_points, decimal=10)


@pytest.mark.parametrize("transform_points", [transform_points_np, transform_points_nb])
def test_wrong_input_shape(transform_points: Callable) -> None:
    tf = np.eye(4)

    with pytest.raises(AssertionError):
        points = np.zeros((3, 10))
        transform_points(points, tf)

    with pytest.raises(AssertionError):
        points = np.zeros((10, 4))  # should be 3D for a 4D matrix
        transform_points(points, tf)

    with pytest.raises(AssertionError):
        points = np.zeros((10, 3))  # should be 2D for a 3D matrix
        transform_points(points, tf[:3, :3])


def test_transform_points_equivalence() -> None:
    """Test equivalence of numpy and numba implementations of transform_points"""

    # Identity transforms and random points for 2D/3D
    tf_3d = np.eye(4)
    tf_2d = np.eye(3)
    pts_3d = np.random.randn(10, 3)
    pts_2d = np.random.randn(10, 2)

    def test_dim_and_equal(pts: np.ndarray, tf: np.ndarray, dims: int) -> None:
        res_np = transform_points_np(pts, tf)
        assert res_np.shape == (pts.shape[0], dims)
        res_nb = transform_points_nb(pts, tf)
        np.testing.assert_equal(res_nb, res_np)

    # Matching points and transforms
    test_dim_and_equal(pts_3d, tf_3d, 3)
    test_dim_and_equal(pts_2d, tf_2d, 2)

    # Too many transform dimensions
    with pytest.raises(AssertionError):
        transform_points_nb(pts_2d, tf_3d)
    with pytest.raises(AssertionError):
        transform_points_np(pts_2d, tf_3d)


@pytest.mark.parametrize("transform_points_subpixel", [transform_points_subpixel_np, transform_points_subpixel_nb])
def test_transform_points_subpixel_equivalence(transform_points_subpixel: Callable) -> None:
    # Generate random input points and an identity transform
    input_points = np.random.rand(10, 2)
    tf = np.identity(3)

    subpixel_points = transform_points_subpixel(input_points, tf)
    assert subpixel_points.dtype == np.int64()
    np.testing.assert_almost_equal(subpixel_points / (2 ** SUBPIXEL_SHIFT), input_points, decimal=1)
