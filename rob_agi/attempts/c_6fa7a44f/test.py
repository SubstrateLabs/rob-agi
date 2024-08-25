import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_6fa7a44f.main import solve_6fa7a44f


def test_6fa7a44f_example_0():
    input_grid = ColoredGrid(values=
[[9, 1, 4], [9, 1, 4], [2, 1, 1]]
    )
    expected = ColoredGrid(values=
[[9, 1, 4], [9, 1, 4], [2, 1, 1], [2, 1, 1], [9, 1, 4], [9, 1, 4]]
)
    actual = solve_6fa7a44f(input_grid)
    assert actual == expected


def test_6fa7a44f_example_1():
    input_grid = ColoredGrid(values=
[[4, 8, 4], [7, 6, 7], [8, 7, 8]]
    )
    expected = ColoredGrid(values=
[[4, 8, 4], [7, 6, 7], [8, 7, 8], [8, 7, 8], [7, 6, 7], [4, 8, 4]]
)
    actual = solve_6fa7a44f(input_grid)
    assert actual == expected


def test_6fa7a44f_example_2():
    input_grid = ColoredGrid(values=
[[7, 7, 7], [9, 5, 5], [5, 1, 7]]
    )
    expected = ColoredGrid(values=
[[7, 7, 7], [9, 5, 5], [5, 1, 7], [5, 1, 7], [9, 5, 5], [7, 7, 7]]
)
    actual = solve_6fa7a44f(input_grid)
    assert actual == expected


def test_6fa7a44f_example_3():
    input_grid = ColoredGrid(values=
[[2, 6, 9], [2, 6, 9], [2, 9, 2]]
    )
    expected = ColoredGrid(values=
[[2, 6, 9], [2, 6, 9], [2, 9, 2], [2, 9, 2], [2, 6, 9], [2, 6, 9]]
)
    actual = solve_6fa7a44f(input_grid)
    assert actual == expected



def test_6fa7a44f_test_case_0():
    input_grid = ColoredGrid(values=
[[2, 9, 2], [8, 5, 2], [2, 2, 8]]
    )
    expected = ColoredGrid(values=
[[2, 9, 2], [8, 5, 2], [2, 2, 8], [2, 2, 8], [8, 5, 2], [2, 9, 2]]
    )
    actual = solve_6fa7a44f(input_grid)
    assert actual == expected

