import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_3979b1a8.main import solve_3979b1a8

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_3979b1a8_example_0():
    input_grid = ColoredGrid(values=
[[9, 8, 8, 8, 9],
 [8, 8, 2, 8, 8],
 [8, 2, 2, 2, 8],
 [8, 8, 2, 8, 8],
 [9, 8, 8, 8, 9]]
    )
    expected = ColoredGrid(values=
[[9, 8, 8, 8, 9, 9, 2, 8, 9, 2],
 [8, 8, 2, 8, 8, 9, 2, 8, 9, 2],
 [8, 2, 2, 2, 8, 9, 2, 8, 9, 2],
 [8, 8, 2, 8, 8, 9, 2, 8, 9, 2],
 [9, 8, 8, 8, 9, 9, 2, 8, 9, 2],
 [9, 9, 9, 9, 9, 2, 2, 8, 9, 2],
 [2, 2, 2, 2, 2, 2, 8, 8, 9, 2],
 [8, 8, 8, 8, 8, 8, 8, 9, 9, 2],
 [9, 9, 9, 9, 9, 9, 9, 9, 2, 2],
 [2, 2, 2, 2, 2, 2, 2, 2, 2, 8]]
)
    actual = solve_3979b1a8(input_grid)
    assert actual == expected


def test_3979b1a8_example_1():
    input_grid = ColoredGrid(values=
[[2, 3, 3, 3, 2],
 [3, 3, 5, 3, 3],
 [3, 5, 5, 5, 3],
 [3, 3, 5, 3, 3],
 [2, 3, 3, 3, 2]]
    )
    expected = ColoredGrid(values=
[[2, 3, 3, 3, 2, 2, 5, 3, 2, 5],
 [3, 3, 5, 3, 3, 2, 5, 3, 2, 5],
 [3, 5, 5, 5, 3, 2, 5, 3, 2, 5],
 [3, 3, 5, 3, 3, 2, 5, 3, 2, 5],
 [2, 3, 3, 3, 2, 2, 5, 3, 2, 5],
 [2, 2, 2, 2, 2, 5, 5, 3, 2, 5],
 [5, 5, 5, 5, 5, 5, 3, 3, 2, 5],
 [3, 3, 3, 3, 3, 3, 3, 2, 2, 5],
 [2, 2, 2, 2, 2, 2, 2, 2, 5, 5],
 [5, 5, 5, 5, 5, 5, 5, 5, 5, 3]]
)
    actual = solve_3979b1a8(input_grid)
    assert actual == expected



