import numpy as np
import pytest
import transforms3d

from l5kit.geometry import SUBPIXEL_SHIFT, transform_point, transform_points, transform_points_subpixel


def test_transform_points() -> None:
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


def test_transform_points_revert_equivalence() -> None:
    input_points = np.random.rand(10, 3)

    #  Generate some random transformation matrix
    tf = np.identity(4)
    tf[:3, :3] = transforms3d.euler.euler2mat(np.random.rand(), np.random.rand(), np.random.rand())
    tf[3, :3] = np.random.rand(3)

    output_points = transform_points(input_points, tf)

    tf_inv = np.linalg.inv(tf)

    input_points_recovered = transform_points(output_points, tf_inv)

    np.testing.assert_almost_equal(input_points_recovered, input_points, decimal=10)


def test_wrong_input_shape() -> None:
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


def test_transform_points_subpixel_equivalence() -> None:
    # Generate random input points and an identity transform
    input_points = np.random.rand(10, 2)
    tf = np.identity(3)

    subpixel_points = transform_points_subpixel(input_points, tf)
    assert subpixel_points.dtype == np.int64()
    np.testing.assert_almost_equal(subpixel_points / (2 ** SUBPIXEL_SHIFT), input_points, decimal=1)
