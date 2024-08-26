import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_4cd1b7b2.main import solve_4cd1b7b2

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_4cd1b7b2_example_0():
    input_grid = ColoredGrid(values=
[[1, 0, 3, 4], [0, 0, 2, 1], [2, 1, 4, 0], [0, 3, 1, 2]]
    )
    expected = ColoredGrid(values=
[[1, 2, 3, 4], [3, 4, 2, 1], [2, 1, 4, 3], [4, 3, 1, 2]]
)
    actual = solve_4cd1b7b2(input_grid)
    assert actual == expected


def test_4cd1b7b2_example_1():
    input_grid = ColoredGrid(values=
[[0, 4, 2, 3], [4, 1, 0, 2], [0, 3, 4, 0], [3, 0, 1, 4]]
    )
    expected = ColoredGrid(values=
[[1, 4, 2, 3], [4, 1, 3, 2], [2, 3, 4, 1], [3, 2, 1, 4]]
)
    actual = solve_4cd1b7b2(input_grid)
    assert actual == expected


def test_4cd1b7b2_example_2():
    input_grid = ColoredGrid(values=
[[3, 0, 2, 1], [1, 0, 0, 0], [4, 3, 0, 2], [0, 1, 4, 3]]
    )
    expected = ColoredGrid(values=
[[3, 4, 2, 1], [1, 2, 3, 4], [4, 3, 1, 2], [2, 1, 4, 3]]
)
    actual = solve_4cd1b7b2(input_grid)
    assert actual == expected



