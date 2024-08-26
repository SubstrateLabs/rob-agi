import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_48f8583b.main import solve_48f8583b

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_48f8583b_example_0():
    input_grid = ColoredGrid(values=
[[9, 9, 6], [3, 8, 8], [8, 3, 3]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 9, 9, 6],
 [0, 0, 0, 0, 0, 0, 3, 8, 8],
 [0, 0, 0, 0, 0, 0, 8, 3, 3],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_48f8583b(input_grid)
    assert actual == expected


def test_48f8583b_example_1():
    input_grid = ColoredGrid(values=
[[8, 5, 5], [8, 8, 8], [5, 9, 9]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 8, 5, 5, 8, 5, 5],
 [0, 0, 0, 8, 8, 8, 8, 8, 8],
 [0, 0, 0, 5, 9, 9, 5, 9, 9]]
)
    actual = solve_48f8583b(input_grid)
    assert actual == expected


def test_48f8583b_example_2():
    input_grid = ColoredGrid(values=
[[7, 1, 7], [1, 7, 7], [7, 1, 7]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 7, 1, 7, 0, 0, 0],
 [0, 0, 0, 1, 7, 7, 0, 0, 0],
 [0, 0, 0, 7, 1, 7, 0, 0, 0],
 [7, 1, 7, 0, 0, 0, 0, 0, 0],
 [1, 7, 7, 0, 0, 0, 0, 0, 0],
 [7, 1, 7, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 7, 1, 7, 0, 0, 0],
 [0, 0, 0, 1, 7, 7, 0, 0, 0],
 [0, 0, 0, 7, 1, 7, 0, 0, 0]]
)
    actual = solve_48f8583b(input_grid)
    assert actual == expected


def test_48f8583b_example_3():
    input_grid = ColoredGrid(values=
[[3, 2, 7], [2, 2, 7], [5, 5, 7]]
    )
    expected = ColoredGrid(values=
[[3, 2, 7, 0, 0, 0, 0, 0, 0],
 [2, 2, 7, 0, 0, 0, 0, 0, 0],
 [5, 5, 7, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_48f8583b(input_grid)
    assert actual == expected


def test_48f8583b_example_4():
    input_grid = ColoredGrid(values=
[[1, 6, 6], [5, 1, 6], [5, 5, 5]]
    )
    expected = ColoredGrid(values=
[[1, 6, 6, 0, 0, 0, 0, 0, 0],
 [5, 1, 6, 0, 0, 0, 0, 0, 0],
 [5, 5, 5, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 6, 6, 0, 0, 0],
 [0, 0, 0, 5, 1, 6, 0, 0, 0],
 [0, 0, 0, 5, 5, 5, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_48f8583b(input_grid)
    assert actual == expected


def test_48f8583b_example_5():
    input_grid = ColoredGrid(values=
[[4, 4, 2], [2, 2, 2], [2, 4, 2]]
    )
    expected = ColoredGrid(values=
[[4, 4, 2, 4, 4, 2, 0, 0, 0],
 [2, 2, 2, 2, 2, 2, 0, 0, 0],
 [2, 4, 2, 2, 4, 2, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 4, 4, 2, 0, 0, 0],
 [0, 0, 0, 2, 2, 2, 0, 0, 0],
 [0, 0, 0, 2, 4, 2, 0, 0, 0]]
)
    actual = solve_48f8583b(input_grid)
    assert actual == expected



