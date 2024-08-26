import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_27a77e38.main import solve_27a77e38

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_27a77e38_example_0():
    input_grid = ColoredGrid(values=
[[2, 2, 3], [5, 5, 5], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 2, 3], [5, 5, 5], [0, 2, 0]]
)
    actual = solve_27a77e38(input_grid)
    assert actual == expected


def test_27a77e38_example_1():
    input_grid = ColoredGrid(values=
[[3, 6, 4, 2, 4],
 [8, 4, 3, 3, 4],
 [5, 5, 5, 5, 5],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[3, 6, 4, 2, 4],
 [8, 4, 3, 3, 4],
 [5, 5, 5, 5, 5],
 [0, 0, 0, 0, 0],
 [0, 0, 4, 0, 0]]
)
    actual = solve_27a77e38(input_grid)
    assert actual == expected


def test_27a77e38_example_2():
    input_grid = ColoredGrid(values=
[[1, 9, 9, 6, 1, 8, 4],
 [4, 6, 7, 8, 9, 7, 1],
 [9, 3, 1, 4, 1, 3, 6],
 [5, 5, 5, 5, 5, 5, 5],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 9, 9, 6, 1, 8, 4],
 [4, 6, 7, 8, 9, 7, 1],
 [9, 3, 1, 4, 1, 3, 6],
 [5, 5, 5, 5, 5, 5, 5],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0]]
)
    actual = solve_27a77e38(input_grid)
    assert actual == expected



