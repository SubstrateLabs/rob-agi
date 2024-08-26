import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_ef26cbf6.main import solve_ef26cbf6

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_ef26cbf6_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0],
 [0, 3, 0, 4, 0, 2, 0, 4, 0, 6, 0],
 [0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0],
 [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
 [1, 0, 0, 4, 0, 1, 0, 4, 1, 0, 1],
 [0, 1, 0, 4, 1, 1, 1, 4, 1, 0, 1],
 [1, 1, 1, 4, 1, 0, 1, 4, 0, 1, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0],
 [0, 3, 0, 4, 0, 2, 0, 4, 0, 6, 0],
 [0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0],
 [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
 [3, 0, 0, 4, 0, 2, 0, 4, 6, 0, 6],
 [0, 3, 0, 4, 2, 2, 2, 4, 6, 0, 6],
 [3, 3, 3, 4, 2, 0, 2, 4, 0, 6, 0]]
)
    actual = solve_ef26cbf6(input_grid)
    assert actual == expected


def test_ef26cbf6_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 4, 1, 0, 0],
 [0, 7, 0, 4, 0, 1, 1],
 [0, 0, 0, 4, 0, 1, 0],
 [4, 4, 4, 4, 4, 4, 4],
 [0, 0, 0, 4, 1, 1, 0],
 [0, 3, 0, 4, 0, 1, 0],
 [0, 0, 0, 4, 1, 1, 1],
 [4, 4, 4, 4, 4, 4, 4],
 [0, 0, 0, 4, 1, 1, 0],
 [0, 8, 0, 4, 0, 1, 1],
 [0, 0, 0, 4, 1, 0, 1]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 4, 7, 0, 0],
 [0, 7, 0, 4, 0, 7, 7],
 [0, 0, 0, 4, 0, 7, 0],
 [4, 4, 4, 4, 4, 4, 4],
 [0, 0, 0, 4, 3, 3, 0],
 [0, 3, 0, 4, 0, 3, 0],
 [0, 0, 0, 4, 3, 3, 3],
 [4, 4, 4, 4, 4, 4, 4],
 [0, 0, 0, 4, 8, 8, 0],
 [0, 8, 0, 4, 0, 8, 8],
 [0, 0, 0, 4, 8, 0, 8]]
)
    actual = solve_ef26cbf6(input_grid)
    assert actual == expected



