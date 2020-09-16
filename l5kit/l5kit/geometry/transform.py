from typing import Sequence, Union, cast
from warnings import warn

import numpy as np
import pymap3d as pm
import transforms3d

# sub-pixel drawing precision constants
SUBPIXEL_SHIFT = 8  # how many bits to shift in drawing
SUBPIXEL_SHIFT_VALUE = 2 ** SUBPIXEL_SHIFT


def compute_agent_pose(agent_centroid_m: np.ndarray, agent_yaw_rad: float) -> np.ndarray:
    """Return the agent pose as a 3x3 matrix. This corresponds to world_from_agent matrix.

    Args:
        agent_centroid_m (np.ndarry): 2D coordinates of the agent
        agent_yaw_rad (float): yaw of the agent

    Returns:
        (np.ndarray): 3x3 world_from_agent matrix
    """
    # Compute agent pose from its position and heading
    return np.array(
        [
            [np.cos(agent_yaw_rad), -np.sin(agent_yaw_rad), agent_centroid_m[0]],
            [np.sin(agent_yaw_rad), np.cos(agent_yaw_rad), agent_centroid_m[1]],
            [0, 0, 1],
        ]
    )


def rotation33_as_yaw(rotation: np.ndarray) -> float:
    """Compute the yaw component of given 3x3 rotation matrix.

    Args:
        rotation (np.ndarray): 3x3 rotation matrix (np.float64 dtype recommended)

    Returns:
        float: yaw rotation in radians
    """
    return cast(float, transforms3d.euler.mat2euler(rotation)[2])


def yaw_as_rotation33(yaw: float) -> np.ndarray:
    """Create a 3x3 rotation matrix from given yaw.
    The rotation is counter-clockwise and it is equivalent to:
    [cos(yaw), -sin(yaw), 0.0],
    [sin(yaw), cos(yaw), 0.0],
    [0.0, 0.0, 1.0],

    Args:
        yaw (float): yaw rotation in radians

    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    return transforms3d.euler.euler2mat(0, 0, yaw)


def flip_y_axis(tm: np.ndarray, y_dim_size: int) -> np.ndarray:
    """Return a new matrix that also performs a flip on the y axis.

    Args:
        tm: the original 3x3 matrix
        y_dim_size: this should match the resolution on y. It makes all coordinates positive

    Returns: a new 3x3 matrix.

    """
    flip_y = np.eye(3)
    flip_y[1, 1] = -1
    tm = np.matmul(flip_y, tm)
    tm[1, 2] += y_dim_size
    return tm


def transform_points(points: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
    """Transform points using transformation matrix.
    Note this function assumes points.shape[1] == matrix.shape[1] - 1, which means that the last
    row in the matrix does not influence the final result.
    For 2D points only the first 2x3 part of the matrix will be used.

    Args:
        points (np.ndarray): Input points (Nx2) or (Nx3).
        transf_matrix (np.ndarray): 3x3 or 4x4 transformation matrix for 2D and 3D input respectively

    Returns:
        np.ndarray: array of shape (N,2) for 2D input points, or (N,3) points for 3D input points
    """
    assert len(points.shape) == len(transf_matrix.shape) == 2, (
        f"dimensions mismatch, both points ({points.shape}) and "
        f"transf_matrix ({transf_matrix.shape}) needs to be 2D numpy ndarrays."
    )
    assert (
        transf_matrix.shape[0] == transf_matrix.shape[1]
    ), f"transf_matrix ({transf_matrix.shape}) should be a square matrix."

    if points.shape[1] not in [2, 3]:
        raise AssertionError("Points input should be (N, 2) or (N, 3) shape, received {}".format(points.shape))

    assert points.shape[1] == transf_matrix.shape[1] - 1, "points dim should be one less than matrix dim"

    num_dims = len(transf_matrix) - 1
    transf_matrix = transf_matrix.T

    return points @ transf_matrix[:num_dims, :num_dims] + transf_matrix[-1, :num_dims]


def transform_points_subpixel(points: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
    """
    Transform points using transformation matrix and cast coordinates to numpy.int but keep fractional part by
    previously multiplying by 2**SUBPIXEL_SHIFT

    Args:
        points (np.ndarray): Input points array of floats of shape (Nx2), (Nx3) or (Nx4).
        transf_matrix (np.ndarray): 3x3 or 4x4 transformation matrix for 2D and 3D input respectively

    Returns:
        np.ndarray: array of int of shape (N,2) for 2D input points, or (N,3) points for 3D input points
    """
    points_subpixel = transform_points(points, transf_matrix) * SUBPIXEL_SHIFT_VALUE
    points_subpixel = points_subpixel.astype(np.int)
    return points_subpixel


def transform_point(point: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
    """Transform a single vector using transformation matrix.

    Args:
        point (np.ndarray): vector of shape (N)
        transf_matrix (np.ndarray): transformation matrix of shape (N+1, N+1)

    Returns:
        np.ndarray: vector of same shape as input point
    """
    point_ext = np.hstack((point, np.ones(1)))
    return np.matmul(transf_matrix, point_ext)[: point.shape[0]]


def ecef_to_geodetic(point: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """Convert given ECEF coordinate into latitude, longitude, altitude.

    Args:
        point (Union[np.ndarray, Sequence[float]]): ECEF coordinate vector

    Returns:
        np.ndarray: latitude, altitude, longitude
    """
    return np.array(pm.ecef2geodetic(point[0], point[1], point[2]))


def geodetic_to_ecef(lla_point: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """Convert given latitude, longitude, and optionally altitude into ECEF
    coordinates. If no altitude is given, altitude 0 is assumed.

    Args:
        lla_point (Union[np.ndarray, Sequence[float]]): Latitude, Longitude and optionally Altitude

    Returns:
        np.ndarray: 3D ECEF coordinate
    """
    if len(lla_point) == 2:
        return np.array(pm.geodetic2ecef(lla_point[0], lla_point[1], 0), dtype=np.float64)
    else:
        return np.array(pm.geodetic2ecef(lla_point[0], lla_point[1], lla_point[2]), dtype=np.float64)


# Numba optimised versions of transform_points routines

try:
    import numba as nb

    @nb.njit
    def _transform_points_nb(points: np.ndarray, transf_matrix: np.ndarray, scale: int, res: np.ndarray) -> None:
        """
        Internal Numba function to transform points with optional scaling and int conversion.

        Note that no checking is done in this function so should be performed by callers (some bounds and input
        checking will be performed but may produce cryptic error messages that are best avoided).

        Args:
            points (np.ndarray): Input points (Nx2), (Nx3) or (Nx4).
            transf_matrix (np.ndarray): 3x3 or 4x4 transformation matrix for 2D and 3D input respectively
            scale (int): Value to scale results by
            res (np.ndarray): The output array into which to output results
        """
        num_dims = transf_matrix.shape[0] - 1
        # For each point compute a dot product with the transformation matrix, fixing the Z coordinate to 1.
        for p in range(points.shape[0]):
            for out_dim in range(num_dims):
                val = 0
                for dim in range(num_dims):
                    val += points[p, dim] * transf_matrix[out_dim, dim]
                val += transf_matrix[out_dim, num_dims]  # Z coordinate fixed to 1
                res[p, out_dim] = val * scale

    def transform_points_nb(points: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
        assert len(points.shape) == len(transf_matrix.shape) == 2
        assert transf_matrix.shape[0] == transf_matrix.shape[1]

        if points.shape[1] not in [2, 3]:
            raise AssertionError("Points input should be (N, 2) or (N,3) shape, received {}".format(points.shape))

        assert points.shape[1] == transf_matrix.shape[1] - 1, "points dim should be one less than matrix dim"

        # If points contains more dimensions than transforms then only transformed dimensions used
        res = np.empty((points.shape[0], transf_matrix.shape[0] - 1), dtype=points.dtype)
        _transform_points_nb(points, transf_matrix, 1, res)
        return res

    def transform_points_subpixel_nb(points: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
        assert len(points.shape) == len(transf_matrix.shape) == 2
        assert transf_matrix.shape[0] == transf_matrix.shape[1]

        if points.shape[1] not in [2, 3]:
            raise AssertionError("Points input should be (N, 2) or (N,3) shape, received {}".format(points.shape))

        assert points.shape[1] == transf_matrix.shape[1] - 1, "points dim should be one less than matrix dim"

        # If points contains more dimensions than transforms then only transformed dimensions used
        res = np.empty((points.shape[0], transf_matrix.shape[0] - 1), dtype=np.int64())
        _transform_points_nb(points, transf_matrix, SUBPIXEL_SHIFT_VALUE, res)
        return res

    # Replace original functions with numba optimised versions
    transform_points_np = transform_points
    transform_points_nb.__doc__ = transform_points_np.__doc__
    transform_points = transform_points_nb
    transform_points_subpixel_np = transform_points_subpixel
    transform_points_subpixel_nb.__doc__ = transform_points_subpixel_np.__doc__
    transform_points_subpixel = transform_points_subpixel_nb

except ImportError:
    pass
except Exception as e:
    warn("Error creating Numba optimised functions. Non-optimised versions will be used.\n" + str(e))
