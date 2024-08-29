import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_25d8a9c8.main import solve_25d8a9c8

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_25d8a9c8_example_0():
    input_grid = ColoredGrid(values=
[[4, 4, 4], [2, 3, 2], [2, 3, 3]]
    )
    expected = ColoredGrid(values=
[[5, 5, 5], [0, 0, 0], [0, 0, 0]]
)
    actual = solve_25d8a9c8(input_grid)
    assert actual == expected


def test_25d8a9c8_example_1():
    input_grid = ColoredGrid(values=
[[7, 3, 3], [6, 6, 6], [3, 7, 7]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [5, 5, 5], [0, 0, 0]]
)
    actual = solve_25d8a9c8(input_grid)
    assert actual == expected


def test_25d8a9c8_example_2():
    input_grid = ColoredGrid(values=
[[2, 9, 2], [4, 4, 4], [9, 9, 9]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [5, 5, 5], [5, 5, 5]]
)
    actual = solve_25d8a9c8(input_grid)
    assert actual == expected


def test_25d8a9c8_example_3():
    input_grid = ColoredGrid(values=
[[2, 2, 4], [2, 2, 4], [1, 1, 1]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [0, 0, 0], [5, 5, 5]]
)
    actual = solve_25d8a9c8(input_grid)
    assert actual == expected



