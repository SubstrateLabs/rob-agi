import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_d631b094.main import solve_d631b094

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_d631b094_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    )
    expected = ColoredGrid(values=
[[1, 1]]
)
    actual = solve_d631b094(input_grid)
    assert actual == expected


def test_d631b094_example_1():
    input_grid = ColoredGrid(values=
[[0, 2, 0], [2, 0, 0], [0, 2, 0]]
    )
    expected = ColoredGrid(values=
[[2, 2, 2]]
)
    actual = solve_d631b094(input_grid)
    assert actual == expected


def test_d631b094_example_2():
    input_grid = ColoredGrid(values=
[[0, 7, 0], [0, 0, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[7]]
)
    actual = solve_d631b094(input_grid)
    assert actual == expected


def test_d631b094_example_3():
    input_grid = ColoredGrid(values=
[[0, 8, 0], [8, 8, 0], [8, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 8, 8, 8]]
)
    actual = solve_d631b094(input_grid)
    assert actual == expected



