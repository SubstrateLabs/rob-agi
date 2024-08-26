import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_fe9372f3.main import solve_fe9372f3

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_fe9372f3_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
 [0, 0, 0, 2, 2, 2, 0, 0, 0, 0],
 [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 0, 0, 0, 4, 0, 0, 0, 1, 0],
 [0, 1, 0, 0, 8, 0, 0, 1, 0, 0],
 [0, 0, 1, 0, 8, 0, 1, 0, 0, 0],
 [0, 0, 0, 1, 2, 1, 0, 0, 0, 0],
 [4, 8, 8, 2, 2, 2, 8, 8, 4, 8],
 [0, 0, 0, 1, 2, 1, 0, 0, 0, 0],
 [0, 0, 1, 0, 8, 0, 1, 0, 0, 0],
 [0, 1, 0, 0, 8, 0, 0, 1, 0, 0],
 [1, 0, 0, 0, 4, 0, 0, 0, 1, 0],
 [0, 0, 0, 0, 8, 0, 0, 0, 0, 1],
 [0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 4, 0, 0, 0, 0, 0]]
)
    actual = solve_fe9372f3(input_grid)
    assert actual == expected


def test_fe9372f3_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 0, 0, 8, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 0, 8, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [8, 8, 2, 2, 2, 8, 8, 4, 8, 8, 4, 8, 8, 4, 8, 8],
 [0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 0, 8, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 8, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_fe9372f3(input_grid)
    assert actual == expected



