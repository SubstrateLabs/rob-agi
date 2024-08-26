import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_e633a9e5.main import solve_e633a9e5

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_e633a9e5_example_0():
    input_grid = ColoredGrid(values=
[[6, 5, 5], [5, 1, 7], [4, 5, 2]]
    )
    expected = ColoredGrid(values=
[[6, 6, 5, 5, 5],
 [6, 6, 5, 5, 5],
 [5, 5, 1, 7, 7],
 [4, 4, 5, 2, 2],
 [4, 4, 5, 2, 2]]
)
    actual = solve_e633a9e5(input_grid)
    assert actual == expected


def test_e633a9e5_example_1():
    input_grid = ColoredGrid(values=
[[1, 3, 5], [1, 2, 8], [8, 3, 8]]
    )
    expected = ColoredGrid(values=
[[1, 1, 3, 5, 5],
 [1, 1, 3, 5, 5],
 [1, 1, 2, 8, 8],
 [8, 8, 3, 8, 8],
 [8, 8, 3, 8, 8]]
)
    actual = solve_e633a9e5(input_grid)
    assert actual == expected


def test_e633a9e5_example_2():
    input_grid = ColoredGrid(values=
[[2, 3, 7], [2, 1, 6], [1, 5, 7]]
    )
    expected = ColoredGrid(values=
[[2, 2, 3, 7, 7],
 [2, 2, 3, 7, 7],
 [2, 2, 1, 6, 6],
 [1, 1, 5, 7, 7],
 [1, 1, 5, 7, 7]]
)
    actual = solve_e633a9e5(input_grid)
    assert actual == expected



