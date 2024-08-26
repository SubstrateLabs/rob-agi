import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_2072aba6.main import solve_2072aba6

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_2072aba6_example_0():
    input_grid = ColoredGrid(values=
[[0, 5, 0], [5, 5, 5], [0, 5, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 1, 2, 0, 0],
 [0, 0, 2, 1, 0, 0],
 [1, 2, 1, 2, 1, 2],
 [2, 1, 2, 1, 2, 1],
 [0, 0, 1, 2, 0, 0],
 [0, 0, 2, 1, 0, 0]]
)
    actual = solve_2072aba6(input_grid)
    assert actual == expected


def test_2072aba6_example_1():
    input_grid = ColoredGrid(values=
[[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    )
    expected = ColoredGrid(values=
[[1, 2, 0, 0, 0, 0],
 [2, 1, 0, 0, 0, 0],
 [0, 0, 1, 2, 0, 0],
 [0, 0, 2, 1, 0, 0],
 [0, 0, 0, 0, 1, 2],
 [0, 0, 0, 0, 2, 1]]
)
    actual = solve_2072aba6(input_grid)
    assert actual == expected


def test_2072aba6_example_2():
    input_grid = ColoredGrid(values=
[[0, 5, 0], [0, 5, 5], [5, 5, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 1, 2, 0, 0],
 [0, 0, 2, 1, 0, 0],
 [0, 0, 1, 2, 1, 2],
 [0, 0, 2, 1, 2, 1],
 [1, 2, 1, 2, 0, 0],
 [2, 1, 2, 1, 0, 0]]
)
    actual = solve_2072aba6(input_grid)
    assert actual == expected



