import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_27f8ce4f.main import solve_27f8ce4f


def test_27f8ce4f_example_0():
    input_grid = ColoredGrid(values=
[[8, 8, 1], [8, 6, 1], [4, 9, 6]]
    )
    expected = ColoredGrid(values=
[[8, 8, 1, 8, 8, 1, 0, 0, 0],
 [8, 6, 1, 8, 6, 1, 0, 0, 0],
 [4, 9, 6, 4, 9, 6, 0, 0, 0],
 [8, 8, 1, 0, 0, 0, 0, 0, 0],
 [8, 6, 1, 0, 0, 0, 0, 0, 0],
 [4, 9, 6, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_27f8ce4f(input_grid)
    assert actual == expected


def test_27f8ce4f_example_1():
    input_grid = ColoredGrid(values=
[[7, 7, 1], [4, 7, 1], [3, 3, 7]]
    )
    expected = ColoredGrid(values=
[[7, 7, 1, 7, 7, 1, 0, 0, 0],
 [4, 7, 1, 4, 7, 1, 0, 0, 0],
 [3, 3, 7, 3, 3, 7, 0, 0, 0],
 [0, 0, 0, 7, 7, 1, 0, 0, 0],
 [0, 0, 0, 4, 7, 1, 0, 0, 0],
 [0, 0, 0, 3, 3, 7, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 7, 7, 1],
 [0, 0, 0, 0, 0, 0, 4, 7, 1],
 [0, 0, 0, 0, 0, 0, 3, 3, 7]]
)
    actual = solve_27f8ce4f(input_grid)
    assert actual == expected


def test_27f8ce4f_example_2():
    input_grid = ColoredGrid(values=
[[4, 5, 4], [2, 2, 5], [5, 5, 4]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 4, 5, 4, 0, 0, 0],
 [0, 0, 0, 2, 2, 5, 0, 0, 0],
 [0, 0, 0, 5, 5, 4, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 4, 5, 4],
 [0, 0, 0, 0, 0, 0, 2, 2, 5],
 [0, 0, 0, 0, 0, 0, 5, 5, 4],
 [4, 5, 4, 4, 5, 4, 0, 0, 0],
 [2, 2, 5, 2, 2, 5, 0, 0, 0],
 [5, 5, 4, 5, 5, 4, 0, 0, 0]]
)
    actual = solve_27f8ce4f(input_grid)
    assert actual == expected


def test_27f8ce4f_example_3():
    input_grid = ColoredGrid(values=
[[1, 2, 3], [9, 9, 1], [2, 9, 4]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 2, 3, 1, 2, 3, 0, 0, 0],
 [9, 9, 1, 9, 9, 1, 0, 0, 0],
 [2, 9, 4, 2, 9, 4, 0, 0, 0],
 [0, 0, 0, 1, 2, 3, 0, 0, 0],
 [0, 0, 0, 9, 9, 1, 0, 0, 0],
 [0, 0, 0, 2, 9, 4, 0, 0, 0]]
)
    actual = solve_27f8ce4f(input_grid)
    assert actual == expected



def test_27f8ce4f_test_case_0():
    input_grid = ColoredGrid(values=
[[9, 6, 7], [8, 7, 7], [2, 8, 7]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 9, 6, 7],
 [0, 0, 0, 0, 0, 0, 8, 7, 7],
 [0, 0, 0, 0, 0, 0, 2, 8, 7],
 [0, 0, 0, 9, 6, 7, 9, 6, 7],
 [0, 0, 0, 8, 7, 7, 8, 7, 7],
 [0, 0, 0, 2, 8, 7, 2, 8, 7],
 [0, 0, 0, 0, 0, 0, 9, 6, 7],
 [0, 0, 0, 0, 0, 0, 8, 7, 7],
 [0, 0, 0, 0, 0, 0, 2, 8, 7]]
    )
    actual = solve_27f8ce4f(input_grid)
    assert actual == expected

