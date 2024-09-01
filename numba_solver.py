from numba import (
    float64,
    int64,
    njit,
    uint8,
)
from numpy import (
    argmin,
    empty,
    full,
    inf,
    ndarray,
)


jit_decorator = njit(
    int64[:, ::1](
        float64[:, :, ::1],
        int64,
        int64[:, ::1],
        float64,
        float64[:, ::1],
        int64,
        uint8[:, ::1],
        uint8[:, ::1],
        int64,
    ),
    cache=True,
    nogil=True,
    fastmath=True
)


def implementation(
    cache: ndarray,
    disparity_levels: int,
    disparity_map: ndarray,
    edge_smoothness: float,
    edges: ndarray,
    height: int,
    left_image: ndarray,
    right_image: ndarray,
    width: int
) -> ndarray:

    for d in range(disparity_levels):
        for d_prime in range(disparity_levels):
            edges[d, d_prime] = edge_smoothness * (d != d_prime)

    x = width - 1
    d = 0
    for y in range(height):
        cache[y, x, d] = abs(
            float(right_image[y, x]) - float(left_image[y, x]))

    for y in range(height):
        for x in range(width - 2, -1, -1):
            for d in range(min(disparity_levels, width - x)):
                best_result = inf
                for d_prime in range(disparity_levels):
                    absolute_difference = abs(
                        float(left_image[y, x + d])
                        - float(right_image[y, x])
                    )
                    loss = (
                        absolute_difference
                        + edges[d, d_prime]
                        + cache[y, x + 1, d_prime]
                    )
                    if loss < best_result:
                        best_result = loss
                cache[y, x, d] = best_result

    x = 0
    for y in range(height):
        disparity_map[y, x] = argmin(cache[y, x])

    for y in range(height):
        for x in range(1, width):
            minimal_penalty = inf
            for d in range(min(disparity_levels, width - x)):
                penalty = edges[disparity_map[y, x - 1], d] + cache[y, x, d]
                if penalty < minimal_penalty:
                    minimal_penalty = penalty
                    disparity_map[y, x] = d

    return disparity_map * 255 // (disparity_levels - 1)


def solve_numba(
    *,
    left_image: ndarray,
    right_image: ndarray,
    max_disparity: int,
    edge_smoothness: float,
    use_numba=False,
) -> ndarray:

    height, width = left_image.shape
    disparity_levels = min(max_disparity + 1, width - 1)
    edges = empty((disparity_levels, disparity_levels), dtype="float64")
    disparity_map = empty((height, width), dtype="int64")
    cache = full((height, width, disparity_levels), inf, dtype="float64")

    if use_numba:
        solve = jit_decorator(implementation)
    else:
        solve = implementation

    return solve(
        cache,
        disparity_levels,
        disparity_map,
        edge_smoothness,
        edges,
        height,
        left_image,
        right_image,
        width,
    ).astype("uint8")
