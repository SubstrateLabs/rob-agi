import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_8e2edd66.main import solve_8e2edd66

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_8e2edd66_example_0():
    input_grid = ColoredGrid(values=
[[9, 9, 0], [0, 0, 9], [0, 9, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 9],
 [0, 0, 0, 0, 0, 0, 9, 9, 0],
 [0, 0, 0, 0, 0, 0, 9, 0, 9],
 [0, 0, 9, 0, 0, 9, 0, 0, 0],
 [9, 9, 0, 9, 9, 0, 0, 0, 0],
 [9, 0, 9, 9, 0, 9, 0, 0, 0],
 [0, 0, 9, 0, 0, 0, 0, 0, 9],
 [9, 9, 0, 0, 0, 0, 9, 9, 0],
 [9, 0, 9, 0, 0, 0, 9, 0, 9]]
)
    actual = solve_8e2edd66(input_grid)
    assert actual == expected


def test_8e2edd66_example_1():
    input_grid = ColoredGrid(values=
[[8, 8, 0], [0, 8, 8], [0, 8, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 8],
 [0, 0, 0, 0, 0, 0, 8, 0, 0],
 [0, 0, 0, 0, 0, 0, 8, 0, 8],
 [0, 0, 8, 0, 0, 0, 0, 0, 0],
 [8, 0, 0, 0, 0, 0, 0, 0, 0],
 [8, 0, 8, 0, 0, 0, 0, 0, 0],
 [0, 0, 8, 0, 0, 0, 0, 0, 8],
 [8, 0, 0, 0, 0, 0, 8, 0, 0],
 [8, 0, 8, 0, 0, 0, 8, 0, 8]]
)
    actual = solve_8e2edd66(input_grid)
    assert actual == expected


def test_8e2edd66_example_2():
    input_grid = ColoredGrid(values=
[[7, 0, 7], [7, 7, 7], [0, 7, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 7, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 7, 0, 7, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 7, 0, 0, 0, 0, 0, 7, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [7, 0, 7, 0, 0, 0, 7, 0, 7]]
)
    actual = solve_8e2edd66(input_grid)
    assert actual == expected



