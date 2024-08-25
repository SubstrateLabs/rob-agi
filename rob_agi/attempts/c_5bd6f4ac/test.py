import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_5bd6f4ac.main import solve_5bd6f4ac


def test_5bd6f4ac_example_0():
    input_grid = ColoredGrid(values=
[[3, 0, 0, 7, 0, 0, 9, 7, 0],
 [8, 4, 0, 6, 6, 0, 4, 8, 4],
 [1, 7, 0, 0, 0, 0, 4, 0, 0],
 [1, 1, 0, 9, 1, 0, 7, 0, 0],
 [0, 0, 0, 0, 7, 7, 0, 0, 0],
 [8, 0, 0, 1, 7, 0, 8, 4, 0],
 [0, 7, 0, 9, 9, 2, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 5, 0, 0],
 [0, 0, 0, 2, 4, 0, 8, 0, 0]]
    )
    expected = ColoredGrid(values=
[[9, 7, 0], [4, 8, 4], [4, 0, 0]]
)
    actual = solve_5bd6f4ac(input_grid)
    assert actual == expected


def test_5bd6f4ac_example_1():
    input_grid = ColoredGrid(values=
[[9, 0, 0, 0, 0, 0, 0, 6, 0],
 [0, 4, 0, 7, 0, 5, 0, 8, 1],
 [0, 2, 0, 0, 7, 1, 4, 4, 5],
 [0, 6, 0, 0, 4, 0, 0, 0, 0],
 [8, 3, 0, 4, 2, 0, 0, 9, 7],
 [0, 0, 2, 3, 0, 2, 0, 6, 7],
 [4, 0, 4, 0, 3, 4, 7, 0, 7],
 [7, 1, 0, 0, 0, 0, 3, 0, 0],
 [3, 2, 0, 0, 4, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 6, 0], [0, 8, 1], [4, 4, 5]]
)
    actual = solve_5bd6f4ac(input_grid)
    assert actual == expected


def test_5bd6f4ac_example_2():
    input_grid = ColoredGrid(values=
[[2, 5, 0, 0, 6, 0, 0, 0, 0],
 [2, 5, 5, 7, 0, 0, 6, 0, 1],
 [0, 3, 0, 0, 0, 1, 9, 4, 0],
 [0, 7, 0, 6, 0, 0, 0, 0, 0],
 [0, 9, 0, 0, 0, 1, 0, 0, 8],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 4, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 4],
 [0, 5, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [6, 0, 1], [9, 4, 0]]
)
    actual = solve_5bd6f4ac(input_grid)
    assert actual == expected


def test_5bd6f4ac_example_3():
    input_grid = ColoredGrid(values=
[[0, 5, 0, 0, 8, 0, 0, 0, 4],
 [0, 0, 0, 0, 0, 0, 3, 0, 0],
 [0, 0, 0, 0, 2, 1, 0, 0, 3],
 [0, 1, 0, 0, 0, 0, 3, 0, 0],
 [1, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 8, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 9, 4, 0, 0, 0, 0, 0],
 [3, 0, 7, 0, 0, 2, 0, 0, 6]]
    )
    expected = ColoredGrid(values=
[[0, 0, 4], [3, 0, 0], [0, 0, 3]]
)
    actual = solve_5bd6f4ac(input_grid)
    assert actual == expected



def test_5bd6f4ac_test_case_0():
    input_grid = ColoredGrid(values=
[[6, 9, 0, 0, 1, 0, 5, 8, 9],
 [2, 9, 0, 6, 0, 8, 0, 9, 0],
 [0, 0, 0, 0, 0, 9, 9, 2, 0],
 [9, 2, 6, 0, 0, 8, 0, 6, 8],
 [7, 7, 4, 0, 7, 0, 9, 0, 0],
 [0, 0, 7, 0, 0, 1, 5, 7, 4],
 [4, 1, 0, 0, 7, 5, 0, 0, 9],
 [9, 9, 0, 0, 0, 0, 1, 0, 0],
 [4, 9, 2, 0, 0, 0, 8, 4, 0]]
    )
    expected = ColoredGrid(values=
[[5, 8, 9], [0, 9, 0], [9, 2, 0]]
    )
    actual = solve_5bd6f4ac(input_grid)
    assert actual == expected

