import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_292dd178.main import solve_292dd178

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_292dd178_example_0():
    input_grid = ColoredGrid(values=
[[8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [8, 1, 1, 8, 1, 8, 8, 8, 8, 8],
 [8, 1, 8, 8, 1, 8, 8, 8, 8, 8],
 [8, 1, 8, 8, 1, 8, 8, 8, 8, 8],
 [8, 1, 1, 1, 1, 8, 8, 8, 8, 8],
 [8, 8, 8, 8, 8, 8, 1, 1, 1, 1],
 [8, 8, 8, 8, 8, 8, 1, 8, 8, 1],
 [8, 8, 8, 8, 8, 8, 1, 8, 8, 1],
 [8, 8, 8, 8, 8, 8, 1, 1, 8, 1],
 [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]]
    )
    expected = ColoredGrid(values=
[[8, 8, 8, 2, 8, 8, 8, 8, 8, 8],
 [8, 8, 8, 2, 8, 8, 8, 8, 8, 8],
 [8, 1, 1, 2, 1, 8, 8, 8, 8, 8],
 [8, 1, 2, 2, 1, 8, 8, 8, 8, 8],
 [8, 1, 2, 2, 1, 8, 8, 8, 8, 8],
 [8, 1, 1, 1, 1, 8, 8, 8, 8, 8],
 [8, 8, 8, 8, 8, 8, 1, 1, 1, 1],
 [8, 8, 8, 8, 8, 8, 1, 2, 2, 1],
 [8, 8, 8, 8, 8, 8, 1, 2, 2, 1],
 [8, 8, 8, 8, 8, 8, 1, 1, 2, 1],
 [8, 8, 8, 8, 8, 8, 8, 8, 2, 8]]
)
    actual = solve_292dd178(input_grid)
    assert actual == expected


def test_292dd178_example_1():
    input_grid = ColoredGrid(values=
[[5, 5, 5, 5, 5, 5, 5, 5, 5],
 [5, 5, 5, 5, 5, 5, 5, 5, 5],
 [5, 5, 1, 1, 1, 1, 5, 5, 5],
 [5, 5, 1, 5, 5, 1, 5, 5, 5],
 [5, 5, 5, 5, 5, 1, 5, 5, 5],
 [5, 5, 1, 1, 1, 1, 5, 5, 5],
 [5, 5, 5, 5, 5, 5, 5, 5, 5]]
    )
    expected = ColoredGrid(values=
[[5, 5, 5, 5, 5, 5, 5, 5, 5],
 [5, 5, 5, 5, 5, 5, 5, 5, 5],
 [5, 5, 1, 1, 1, 1, 5, 5, 5],
 [5, 5, 1, 2, 2, 1, 5, 5, 5],
 [2, 2, 2, 2, 2, 1, 5, 5, 5],
 [5, 5, 1, 1, 1, 1, 5, 5, 5],
 [5, 5, 5, 5, 5, 5, 5, 5, 5]]
)
    actual = solve_292dd178(input_grid)
    assert actual == expected


def test_292dd178_example_2():
    input_grid = ColoredGrid(values=
[[9, 1, 9, 1, 1, 9, 9, 9, 9],
 [9, 1, 9, 9, 1, 9, 9, 9, 9],
 [9, 1, 9, 9, 1, 9, 9, 9, 9],
 [9, 1, 1, 1, 1, 9, 9, 9, 9],
 [9, 9, 9, 9, 9, 9, 9, 9, 9],
 [9, 9, 9, 9, 9, 9, 9, 9, 9],
 [9, 9, 1, 1, 1, 1, 9, 9, 9],
 [9, 9, 1, 9, 9, 1, 9, 9, 9],
 [9, 9, 1, 9, 9, 9, 9, 9, 9],
 [9, 9, 1, 1, 1, 1, 9, 9, 9],
 [9, 9, 9, 9, 9, 9, 9, 9, 9]]
    )
    expected = ColoredGrid(values=
[[9, 1, 2, 1, 1, 9, 9, 9, 9],
 [9, 1, 2, 2, 1, 9, 9, 9, 9],
 [9, 1, 2, 2, 1, 9, 9, 9, 9],
 [9, 1, 1, 1, 1, 9, 9, 9, 9],
 [9, 9, 9, 9, 9, 9, 9, 9, 9],
 [9, 9, 9, 9, 9, 9, 9, 9, 9],
 [9, 9, 1, 1, 1, 1, 9, 9, 9],
 [9, 9, 1, 2, 2, 1, 9, 9, 9],
 [9, 9, 1, 2, 2, 2, 2, 2, 2],
 [9, 9, 1, 1, 1, 1, 9, 9, 9],
 [9, 9, 9, 9, 9, 9, 9, 9, 9]]
)
    actual = solve_292dd178(input_grid)
    assert actual == expected



