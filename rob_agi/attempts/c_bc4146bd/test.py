import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_bc4146bd.main import solve_bc4146bd

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_bc4146bd_example_0():
    input_grid = ColoredGrid(values=
[[2, 2, 2, 2], [8, 2, 2, 2], [2, 2, 8, 2], [8, 2, 8, 8]]
    )
    expected = ColoredGrid(values=
[[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
 [8, 2, 2, 2, 2, 2, 2, 8, 8, 2, 2, 2, 2, 2, 2, 8, 8, 2, 2, 2],
 [2, 2, 8, 2, 2, 8, 2, 2, 2, 2, 8, 2, 2, 8, 2, 2, 2, 2, 8, 2],
 [8, 2, 8, 8, 8, 8, 2, 8, 8, 2, 8, 8, 8, 8, 2, 8, 8, 2, 8, 8]]
)
    actual = solve_bc4146bd(input_grid)
    assert actual == expected


def test_bc4146bd_example_1():
    input_grid = ColoredGrid(values=
[[9, 5, 1, 5], [1, 5, 9, 1], [9, 1, 5, 5], [5, 5, 5, 1]]
    )
    expected = ColoredGrid(values=
[[9, 5, 1, 5, 5, 1, 5, 9, 9, 5, 1, 5, 5, 1, 5, 9, 9, 5, 1, 5],
 [1, 5, 9, 1, 1, 9, 5, 1, 1, 5, 9, 1, 1, 9, 5, 1, 1, 5, 9, 1],
 [9, 1, 5, 5, 5, 5, 1, 9, 9, 1, 5, 5, 5, 5, 1, 9, 9, 1, 5, 5],
 [5, 5, 5, 1, 1, 5, 5, 5, 5, 5, 5, 1, 1, 5, 5, 5, 5, 5, 5, 1]]
)
    actual = solve_bc4146bd(input_grid)
    assert actual == expected


def test_bc4146bd_example_2():
    input_grid = ColoredGrid(values=
[[5, 5, 2, 5], [2, 3, 3, 2], [5, 2, 5, 3], [3, 5, 3, 2]]
    )
    expected = ColoredGrid(values=
[[5, 5, 2, 5, 5, 2, 5, 5, 5, 5, 2, 5, 5, 2, 5, 5, 5, 5, 2, 5],
 [2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2],
 [5, 2, 5, 3, 3, 5, 2, 5, 5, 2, 5, 3, 3, 5, 2, 5, 5, 2, 5, 3],
 [3, 5, 3, 2, 2, 3, 5, 3, 3, 5, 3, 2, 2, 3, 5, 3, 3, 5, 3, 2]]
)
    actual = solve_bc4146bd(input_grid)
    assert actual == expected


def test_bc4146bd_example_3():
    input_grid = ColoredGrid(values=
[[4, 1, 1, 4], [7, 7, 4, 7], [1, 4, 1, 1], [4, 1, 1, 1]]
    )
    expected = ColoredGrid(values=
[[4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4],
 [7, 7, 4, 7, 7, 4, 7, 7, 7, 7, 4, 7, 7, 4, 7, 7, 7, 7, 4, 7],
 [1, 4, 1, 1, 1, 1, 4, 1, 1, 4, 1, 1, 1, 1, 4, 1, 1, 4, 1, 1],
 [4, 1, 1, 1, 1, 1, 1, 4, 4, 1, 1, 1, 1, 1, 1, 4, 4, 1, 1, 1]]
)
    actual = solve_bc4146bd(input_grid)
    assert actual == expected



