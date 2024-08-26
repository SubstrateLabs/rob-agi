import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_833dafe3.main import solve_833dafe3

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_833dafe3_example_0():
    input_grid = ColoredGrid(values=
[[3, 0, 0], [3, 4, 0], [3, 4, 2]]
    )
    expected = ColoredGrid(values=
[[2, 4, 3, 3, 4, 2],
 [0, 4, 3, 3, 4, 0],
 [0, 0, 3, 3, 0, 0],
 [0, 0, 3, 3, 0, 0],
 [0, 4, 3, 3, 4, 0],
 [2, 4, 3, 3, 4, 2]]
)
    actual = solve_833dafe3(input_grid)
    assert actual == expected


def test_833dafe3_example_1():
    input_grid = ColoredGrid(values=
[[0, 6, 0, 0], [4, 6, 0, 3], [4, 6, 3, 0], [4, 3, 3, 0]]
    )
    expected = ColoredGrid(values=
[[0, 3, 3, 4, 4, 3, 3, 0],
 [0, 3, 6, 4, 4, 6, 3, 0],
 [3, 0, 6, 4, 4, 6, 0, 3],
 [0, 0, 6, 0, 0, 6, 0, 0],
 [0, 0, 6, 0, 0, 6, 0, 0],
 [3, 0, 6, 4, 4, 6, 0, 3],
 [0, 3, 6, 4, 4, 6, 3, 0],
 [0, 3, 3, 4, 4, 3, 3, 0]]
)
    actual = solve_833dafe3(input_grid)
    assert actual == expected



