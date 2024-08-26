import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_6ea4a07e.main import solve_6ea4a07e

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_6ea4a07e_example_0():
    input_grid = ColoredGrid(values=
[[8, 0, 0], [0, 8, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 2, 2], [2, 0, 2], [2, 2, 2]]
)
    actual = solve_6ea4a07e(input_grid)
    assert actual == expected


def test_6ea4a07e_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 3], [0, 3, 0], [3, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 1, 0], [1, 0, 1], [0, 1, 1]]
)
    actual = solve_6ea4a07e(input_grid)
    assert actual == expected


def test_6ea4a07e_example_2():
    input_grid = ColoredGrid(values=
[[5, 0, 0], [5, 5, 0], [5, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 4, 4], [0, 0, 4], [0, 4, 4]]
)
    actual = solve_6ea4a07e(input_grid)
    assert actual == expected


def test_6ea4a07e_example_3():
    input_grid = ColoredGrid(values=
[[5, 5, 5], [0, 0, 5], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [4, 4, 0], [4, 4, 4]]
)
    actual = solve_6ea4a07e(input_grid)
    assert actual == expected


def test_6ea4a07e_example_4():
    input_grid = ColoredGrid(values=
[[0, 8, 0], [0, 8, 0], [8, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 0, 2], [2, 0, 2], [0, 2, 2]]
)
    actual = solve_6ea4a07e(input_grid)
    assert actual == expected


def test_6ea4a07e_example_5():
    input_grid = ColoredGrid(values=
[[8, 0, 8], [0, 8, 0], [0, 8, 0]]
    )
    expected = ColoredGrid(values=
[[0, 2, 0], [2, 0, 2], [2, 0, 2]]
)
    actual = solve_6ea4a07e(input_grid)
    assert actual == expected



