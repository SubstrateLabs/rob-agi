import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_54d9e175.main import solve_54d9e175


def test_54d9e175_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
 [0, 1, 0, 5, 0, 2, 0, 5, 0, 1, 0],
 [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[6, 6, 6, 5, 7, 7, 7, 5, 6, 6, 6],
 [6, 6, 6, 5, 7, 7, 7, 5, 6, 6, 6],
 [6, 6, 6, 5, 7, 7, 7, 5, 6, 6, 6]]
)
    actual = solve_54d9e175(input_grid)
    assert actual == expected


def test_54d9e175_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
 [0, 2, 0, 5, 0, 3, 0, 5, 0, 1, 0],
 [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[7, 7, 7, 5, 8, 8, 8, 5, 6, 6, 6],
 [7, 7, 7, 5, 8, 8, 8, 5, 6, 6, 6],
 [7, 7, 7, 5, 8, 8, 8, 5, 6, 6, 6]]
)
    actual = solve_54d9e175(input_grid)
    assert actual == expected


def test_54d9e175_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
 [0, 3, 0, 5, 0, 1, 0, 5, 0, 4, 0],
 [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 8, 8, 5, 6, 6, 6, 5, 9, 9, 9],
 [8, 8, 8, 5, 6, 6, 6, 5, 9, 9, 9],
 [8, 8, 8, 5, 6, 6, 6, 5, 9, 9, 9]]
)
    actual = solve_54d9e175(input_grid)
    assert actual == expected


def test_54d9e175_example_3():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
 [0, 4, 0, 5, 0, 1, 0, 5, 0, 2, 0],
 [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
 [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
 [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
 [0, 2, 0, 5, 0, 3, 0, 5, 0, 4, 0],
 [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[9, 9, 9, 5, 6, 6, 6, 5, 7, 7, 7],
 [9, 9, 9, 5, 6, 6, 6, 5, 7, 7, 7],
 [9, 9, 9, 5, 6, 6, 6, 5, 7, 7, 7],
 [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
 [7, 7, 7, 5, 8, 8, 8, 5, 9, 9, 9],
 [7, 7, 7, 5, 8, 8, 8, 5, 9, 9, 9],
 [7, 7, 7, 5, 8, 8, 8, 5, 9, 9, 9]]
)
    actual = solve_54d9e175(input_grid)
    assert actual == expected



def test_54d9e175_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
 [0, 2, 0, 5, 0, 3, 0, 5, 0, 4, 0],
 [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
 [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
 [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
 [0, 1, 0, 5, 0, 1, 0, 5, 0, 3, 0],
 [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[7, 7, 7, 5, 8, 8, 8, 5, 9, 9, 9],
 [7, 7, 7, 5, 8, 8, 8, 5, 9, 9, 9],
 [7, 7, 7, 5, 8, 8, 8, 5, 9, 9, 9],
 [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
 [6, 6, 6, 5, 6, 6, 6, 5, 8, 8, 8],
 [6, 6, 6, 5, 6, 6, 6, 5, 8, 8, 8],
 [6, 6, 6, 5, 6, 6, 6, 5, 8, 8, 8]]
    )
    actual = solve_54d9e175(input_grid)
    assert actual == expected

