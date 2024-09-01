from argparse import (
    ArgumentParser,
    FileType,
)
from enum import Enum
from typing import (
    NamedTuple,
    Callable,
)

from PIL import (
    Image,
    ImageOps,
)
from numpy import array

from numba_solver import solve_numba
from numpy_solver import solve_numpy


class Solver(str, Enum):
    NUMPY = "numpy"
    NAIVE = "naive"
    NUMBA = "numba"


class ProblemInput(NamedTuple):
    left_image_name: str
    right_image_name: str
    output_image_name: str
    max_disparity: int
    edge_smoothness: float
    solver: Solver


def read_input() -> ProblemInput:
    parser = ArgumentParser(description="Stereo Vision")
    parser.add_argument(
        "-l", "--left",
        type=FileType("r"),
        help="Path to the left image",
        required=True,
    )
    parser.add_argument(
        "-r", "--right",
        type=FileType("r"),
        help="Path to the right image",
        required=True,
    )
    parser.add_argument(
        "-o", "--output",
        type=FileType("w"),
        help="Resulting disparity map",
        required=True,
    )
    parser.add_argument(
        "-d", "--max-disparity",
        type=int,
        help="Maximal disparity level",
        required=True,
    )
    parser.add_argument(
        "-e", "--edge-smoothness",
        type=float,
        required=True,
    )
    parser.add_argument(
        "-s", "--solver",
        type=Solver,
        choices=[Solver.NAIVE.value, Solver.NUMBA.value, Solver.NUMPY.value],
        help="Choose the solver method: numpy, numba, or naive",
        required=True,
    )
    arguments = parser.parse_args()

    arguments.left.close()
    arguments.right.close()
    arguments.output.close()

    return ProblemInput(
        left_image_name=arguments.left.name,
        right_image_name=arguments.right.name,
        output_image_name=arguments.output.name,
        max_disparity=arguments.max_disparity,
        edge_smoothness=arguments.edge_smoothness,
        solver=arguments.solver,
    )

def solve(
    *,
    problem_input: ProblemInput,
    solver: Callable,
) -> Image.Image:
    left_image = array(ImageOps.grayscale(Image.open(
        problem_input.left_image_name)))
    right_image = array(ImageOps.grayscale(Image.open(
        problem_input.right_image_name)))

    if problem_input.solver == Solver.NAIVE:
        solver = solve_numba
    elif problem_input.solver == Solver.NUMPY:
        solver = solve_numpy
    elif problem_input.solver == Solver.NUMBA:
        solver = (
            lambda *args, **kwargs:
            solve_numba(use_numba=True, *args, **kwargs)
        )

    result = solver(
        left_image=left_image,
        right_image=right_image,
        max_disparity=problem_input.max_disparity,
        edge_smoothness=problem_input.edge_smoothness,
    )

    return Image.fromarray(result)


def main():
    problem_input = read_input()
    disparity_map = solve(problem_input=problem_input, solver=solve_numba)
    disparity_map.save(problem_input.output_image_name)


if __name__ == "__main__":
    main()
