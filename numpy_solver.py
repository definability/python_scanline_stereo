from numpy import (
    absolute,
    add,
    arange,
    argmin,
    array,
    empty,
    fill_diagonal,
    full,
    inf,
    newaxis,
    not_equal,
    subtract,
    uint8,
)
from PIL import Image


def solve_numpy(
    *,
    left_image: Image,
    right_image: Image,
    max_disparity: int,
    edge_smoothness: float,
):

    left_image = array(left_image, dtype=float)
    right_image = array(right_image, dtype=float)

    height = left_image.shape[0]
    width = left_image.shape[1]
    input_disparity_levels = min(max_disparity + 1, width - 1)
    edges = full(
        (input_disparity_levels, input_disparity_levels),
        edge_smoothness,
        dtype=float,
    )
    fill_diagonal(edges, 0)
    disparity_map = empty(left_image.shape, dtype=int)
    cache = full((*left_image.shape, input_disparity_levels), inf)

    penalties_sum = empty(
        (height, input_disparity_levels, input_disparity_levels),
        dtype=float,
    )
    penalties_min = empty((height, input_disparity_levels), dtype=float)

    x = width - 1
    current_disparity = 0
    subtract(
        left_image[:, x], right_image[:, x],
        out=cache[:, x, current_disparity],
    )
    absolute(
        cache[:, x, current_disparity],
        out=cache[:, x, current_disparity],
    )
    for x in range(width - 2, -1, -1):
        disparity_levels = min(input_disparity_levels, width - x)
        add(
            cache[:, x + 1, :disparity_levels, newaxis],
            edges[newaxis, :disparity_levels, :disparity_levels],
            out=penalties_sum[:, :disparity_levels, :disparity_levels],
        )
        penalties_sum[:, :disparity_levels, :disparity_levels].min(
            axis=-2,
            out=penalties_min[:, :disparity_levels],
        )
        subtract(
            left_image[:, x:x + disparity_levels],
            right_image[:, x, newaxis],
            out=cache[:, x, :disparity_levels],
        )
        absolute(
            cache[:, x, :disparity_levels],
            out=cache[:, x, :disparity_levels],
        )
        cache[:, x, :disparity_levels] += penalties_min[:, :disparity_levels]

    x = 0
    argmin(cache[:, x], axis=-1, out=disparity_map[:, x])
    edges = full((height, input_disparity_levels), inf, dtype=float)
    disparities_array = arange(input_disparity_levels)
    for x in range(1, width):
        disparity_levels = min(input_disparity_levels, width - x)
        not_equal(
            disparity_map[:, x - 1, newaxis],
            disparities_array[newaxis, :disparity_levels],
            out=edges[:, :disparity_levels],
        )
        edges[:, :disparity_levels] *= edge_smoothness
        argmin(
            cache[:, x, :disparity_levels]
            + edges[:, :disparity_levels],
            axis=-1,
            out=disparity_map[:, x],
        )

    return (disparity_map * 255 // (input_disparity_levels - 1)).astype(uint8)
