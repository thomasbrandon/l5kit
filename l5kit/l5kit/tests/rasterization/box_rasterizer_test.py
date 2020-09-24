from typing import Callable

import numpy as np
import pytest

from l5kit.data import AGENT_DTYPE, ChunkedDataset, LocalDataManager, filter_agents_by_frames
from l5kit.rasterization import build_rasterizer
from l5kit.rasterization.box_rasterizer import create_boxes_nb, create_boxes_np, draw_boxes


@pytest.mark.parametrize("create_boxes", [create_boxes_np, create_boxes_nb])
def test_create_boxes_simple(create_boxes: Callable) -> None:
    field_dtypes = {name: dtype for (name, dtype, *_) in AGENT_DTYPE}
    centroids = np.array([[20, 20], [60, 60]], dtype=field_dtypes["centroid"])
    extents = np.array([[10, 20]] * 2, dtype=field_dtypes["extent"])
    yaws = np.radians(np.array([0, 90]), dtype=field_dtypes["yaw"])

    exp_boxes = np.array([[(15, 10), (15, 30), (25, 30), (25, 10)], [(70, 55), (50, 55), (70, 65), (50, 65)]])
    boxes = create_boxes(centroids, extents, yaws)

    # Don't want to test order of points so sort points first
    # This may allow some orderings that fail to rasterize properly but hard otherwise.
    # Converting to single value and getting sorted indexes is the easiest way to sort
    def sort_points(arr: np.ndarray) -> np.ndarray:
        return arr[np.argsort(arr[:, 0] + arr[:, 1] / 1000)]

    for box, exp_box in zip(boxes, exp_boxes):
        np.testing.assert_almost_equal(sort_points(box), sort_points(exp_box), decimal=5)


def test_empty_boxes() -> None:
    # naive test with empty arrays
    agents = np.empty(0, dtype=AGENT_DTYPE)
    to_image_space = np.eye(3)
    im = draw_boxes((200, 200), to_image_space, agents, color=255)
    assert im.sum() == 0


def test_draw_boxes() -> None:
    centroid_1 = (90, 100)
    centroid_2 = (150, 160)

    agents = np.zeros(2, dtype=AGENT_DTYPE)
    agents[0]["extent"] = (20, 20, 20)
    agents[0]["centroid"] = centroid_1

    agents[1]["extent"] = (20, 20, 20)
    agents[1]["centroid"] = centroid_2

    to_image_space = np.eye(3)
    im = draw_boxes((200, 200), to_image_space, agents, color=1)

    # due to subpixel precision we can't check the exact number of pixels
    # check that a 10x10 centred on the boxes is all 1
    assert np.allclose(im[centroid_1[1] - 5 : centroid_1[1] + 5, centroid_1[0] - 5 : centroid_1[0] + 5], 1)
    assert np.allclose(im[centroid_2[1] - 5 : centroid_2[1] + 5, centroid_2[0] - 5 : centroid_2[0] + 5], 1)


@pytest.fixture(scope="module")
def hist_data(zarr_dataset: ChunkedDataset) -> tuple:
    hist_frames = zarr_dataset.frames[100:111][::-1]  # reverse to get them as history
    hist_agents = filter_agents_by_frames(hist_frames, zarr_dataset.agents)
    return hist_frames, hist_agents


@pytest.mark.parametrize("ego_center", [(0.5, 0.5), (0.25, 0.5), (0.75, 0.5), (0.5, 0.25), (0.5, 0.75)])
def test_ego_layer_out_center_configs(ego_center: tuple, hist_data: tuple, dmg: LocalDataManager, cfg: dict) -> None:
    cfg["raster_params"]["map_type"] = "box_debug"
    cfg["raster_params"]["ego_center"] = np.asarray(ego_center)

    rasterizer = build_rasterizer(cfg, dmg)
    out = rasterizer.rasterize(hist_data[0][:1], hist_data[1][:1], [])
    assert out[..., -1].sum() > 0


def test_agents_layer_out(hist_data: tuple, dmg: LocalDataManager, cfg: dict) -> None:
    cfg["raster_params"]["map_type"] = "box_debug"

    cfg["raster_params"]["filter_agents_threshold"] = 1.0
    rasterizer = build_rasterizer(cfg, dmg)

    out = rasterizer.rasterize(hist_data[0][:1], hist_data[1][:1], [])
    assert out[..., 0].sum() == 0

    cfg["raster_params"]["filter_agents_threshold"] = 0.0
    rasterizer = build_rasterizer(cfg, dmg)

    out = rasterizer.rasterize(hist_data[0][:1], hist_data[1][:1], [])
    assert out[..., 0].sum() > 0


def test_agent_as_ego(hist_data: tuple, dmg: LocalDataManager, cfg: dict) -> None:
    cfg["raster_params"]["map_type"] = "box_debug"
    cfg["raster_params"]["filter_agents_threshold"] = -1  # take everything
    rasterizer = build_rasterizer(cfg, dmg)

    agents = hist_data[1][0]
    for ag in agents:
        out = rasterizer.rasterize(hist_data[0][:1], hist_data[1][:1], [], ag)
        assert out[..., -1].sum() > 0


def test_out_shape(hist_data: tuple, dmg: LocalDataManager, cfg: dict) -> None:
    hist_length = 5
    cfg["raster_params"]["map_type"] = "box_debug"
    cfg["model_params"]["history_num_frames"] = hist_length

    rasterizer = build_rasterizer(cfg, dmg)

    out = rasterizer.rasterize(hist_data[0][: hist_length + 1], hist_data[1][: hist_length + 1], [])
    assert out.shape == (224, 224, (hist_length + 1) * 2)
