import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_b6afb2da.main import solve_b6afb2da

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_b6afb2da_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 5, 5, 5, 5, 0, 0, 0, 0, 0],
 [0, 5, 5, 5, 5, 0, 0, 0, 0, 0],
 [0, 5, 5, 5, 5, 0, 0, 0, 0, 0],
 [0, 5, 5, 5, 5, 0, 5, 5, 5, 5],
 [0, 0, 0, 0, 0, 0, 5, 5, 5, 5],
 [0, 0, 0, 0, 0, 0, 5, 5, 5, 5],
 [0, 0, 0, 0, 0, 0, 5, 5, 5, 5],
 [0, 0, 0, 0, 0, 0, 5, 5, 5, 5]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 4, 4, 1, 0, 0, 0, 0, 0],
 [0, 4, 2, 2, 4, 0, 0, 0, 0, 0],
 [0, 4, 2, 2, 4, 0, 0, 0, 0, 0],
 [0, 1, 4, 4, 1, 0, 1, 4, 4, 1],
 [0, 0, 0, 0, 0, 0, 4, 2, 2, 4],
 [0, 0, 0, 0, 0, 0, 4, 2, 2, 4],
 [0, 0, 0, 0, 0, 0, 4, 2, 2, 4],
 [0, 0, 0, 0, 0, 0, 1, 4, 4, 1]]
)
    actual = solve_b6afb2da(input_grid)
    assert actual == expected


def test_b6afb2da_example_1():
    input_grid = ColoredGrid(values=
[[5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
 [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
 [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
 [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
 [5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 5, 5, 5, 5, 5, 5],
 [0, 0, 0, 0, 5, 5, 5, 5, 5, 5],
 [0, 0, 0, 0, 5, 5, 5, 5, 5, 5],
 [0, 0, 0, 0, 5, 5, 5, 5, 5, 5]]
    )
    expected = ColoredGrid(values=
[[1, 4, 4, 4, 4, 1, 0, 0, 0, 0],
 [4, 2, 2, 2, 2, 4, 0, 0, 0, 0],
 [4, 2, 2, 2, 2, 4, 0, 0, 0, 0],
 [4, 2, 2, 2, 2, 4, 0, 0, 0, 0],
 [1, 4, 4, 4, 4, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 4, 4, 4, 4, 1],
 [0, 0, 0, 0, 4, 2, 2, 2, 2, 4],
 [0, 0, 0, 0, 4, 2, 2, 2, 2, 4],
 [0, 0, 0, 0, 1, 4, 4, 4, 4, 1]]
)
    actual = solve_b6afb2da(input_grid)
    assert actual == expected



