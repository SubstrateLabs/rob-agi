import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_7953d61e.main import solve_7953d61e

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_7953d61e_example_0():
    input_grid = ColoredGrid(values=
[[4, 1, 9, 1], [1, 9, 1, 4], [9, 1, 4, 6], [4, 1, 6, 6]]
    )
    expected = ColoredGrid(values=
[[4, 1, 9, 1, 1, 4, 6, 6],
 [1, 9, 1, 4, 9, 1, 4, 6],
 [9, 1, 4, 6, 1, 9, 1, 1],
 [4, 1, 6, 6, 4, 1, 9, 4],
 [6, 6, 1, 4, 4, 9, 1, 4],
 [6, 4, 1, 9, 1, 1, 9, 1],
 [4, 1, 9, 1, 6, 4, 1, 9],
 [1, 9, 1, 4, 6, 6, 4, 1]]
)
    actual = solve_7953d61e(input_grid)
    assert actual == expected


def test_7953d61e_example_1():
    input_grid = ColoredGrid(values=
[[6, 2, 6, 2], [6, 6, 5, 5], [1, 1, 1, 2], [5, 1, 2, 1]]
    )
    expected = ColoredGrid(values=
[[6, 2, 6, 2, 2, 5, 2, 1],
 [6, 6, 5, 5, 6, 5, 1, 2],
 [1, 1, 1, 2, 2, 6, 1, 1],
 [5, 1, 2, 1, 6, 6, 1, 5],
 [1, 2, 1, 5, 5, 1, 6, 6],
 [2, 1, 1, 1, 1, 1, 6, 2],
 [5, 5, 6, 6, 2, 1, 5, 6],
 [2, 6, 2, 6, 1, 2, 5, 2]]
)
    actual = solve_7953d61e(input_grid)
    assert actual == expected


def test_7953d61e_example_2():
    input_grid = ColoredGrid(values=
[[6, 7, 7, 6], [7, 1, 6, 6], [9, 1, 6, 6], [9, 1, 6, 1]]
    )
    expected = ColoredGrid(values=
[[6, 7, 7, 6, 6, 6, 6, 1],
 [7, 1, 6, 6, 7, 6, 6, 6],
 [9, 1, 6, 6, 7, 1, 1, 1],
 [9, 1, 6, 1, 6, 7, 9, 9],
 [1, 6, 1, 9, 9, 9, 7, 6],
 [6, 6, 1, 9, 1, 1, 1, 7],
 [6, 6, 1, 7, 6, 6, 6, 7],
 [6, 7, 7, 6, 1, 6, 6, 6]]
)
    actual = solve_7953d61e(input_grid)
    assert actual == expected


def test_7953d61e_example_3():
    input_grid = ColoredGrid(values=
[[4, 9, 1, 8], [8, 4, 1, 8], [4, 8, 8, 1], [1, 1, 1, 8]]
    )
    expected = ColoredGrid(values=
[[4, 9, 1, 8, 8, 8, 1, 8],
 [8, 4, 1, 8, 1, 1, 8, 1],
 [4, 8, 8, 1, 9, 4, 8, 1],
 [1, 1, 1, 8, 4, 8, 4, 1],
 [8, 1, 1, 1, 1, 4, 8, 4],
 [1, 8, 8, 4, 1, 8, 4, 9],
 [8, 1, 4, 8, 1, 8, 1, 1],
 [8, 1, 9, 4, 8, 1, 8, 8]]
)
    actual = solve_7953d61e(input_grid)
    assert actual == expected


def test_7953d61e_example_4():
    input_grid = ColoredGrid(values=
[[1, 1, 2, 1], [6, 6, 7, 6], [7, 6, 2, 1], [1, 6, 2, 6]]
    )
    expected = ColoredGrid(values=
[[1, 1, 2, 1, 1, 6, 1, 6],
 [6, 6, 7, 6, 2, 7, 2, 2],
 [7, 6, 2, 1, 1, 6, 6, 6],
 [1, 6, 2, 6, 1, 6, 7, 1],
 [6, 2, 6, 1, 1, 7, 6, 1],
 [1, 2, 6, 7, 6, 6, 6, 1],
 [6, 7, 6, 6, 2, 2, 7, 2],
 [1, 2, 1, 1, 6, 1, 6, 1]]
)
    actual = solve_7953d61e(input_grid)
    assert actual == expected



