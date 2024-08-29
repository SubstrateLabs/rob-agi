import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_6150a2bd.main import solve_6150a2bd

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_6150a2bd_example_0():
    input_grid = ColoredGrid(values=
[[3, 3, 8], [3, 7, 0], [5, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 5], [0, 7, 3], [8, 3, 3]]
)
    actual = solve_6150a2bd(input_grid)
    assert actual == expected


def test_6150a2bd_example_1():
    input_grid = ColoredGrid(values=
[[5, 5, 2], [1, 0, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [0, 0, 1], [2, 5, 5]]
)
    actual = solve_6150a2bd(input_grid)
    assert actual == expected



