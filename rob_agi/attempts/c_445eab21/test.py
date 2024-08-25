import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_445eab21.main import solve_445eab21


def test_445eab21_example_0():
    input_grid = ColoredGrid(values=
[[0, 7, 7, 7, 7, 0, 0, 0, 0, 0],
 [0, 7, 0, 0, 7, 0, 0, 0, 0, 0],
 [0, 7, 0, 0, 7, 0, 0, 0, 0, 0],
 [0, 7, 7, 7, 7, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 8, 8, 8, 8, 8, 0, 0],
 [0, 0, 0, 8, 0, 0, 0, 8, 0, 0],
 [0, 0, 0, 8, 0, 0, 0, 8, 0, 0],
 [0, 0, 0, 8, 8, 8, 8, 8, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 8], [8, 8]]
)
    actual = solve_445eab21(input_grid)
    assert actual == expected


def test_445eab21_example_1():
    input_grid = ColoredGrid(values=
[[6, 6, 6, 6, 6, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 6, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 6, 0, 0, 0, 0, 0],
 [6, 6, 6, 6, 6, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 7, 7, 7, 7, 7, 7, 0, 0],
 [0, 0, 7, 0, 0, 0, 0, 7, 0, 0],
 [0, 0, 7, 0, 0, 0, 0, 7, 0, 0],
 [0, 0, 7, 7, 7, 7, 7, 7, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[7, 7], [7, 7]]
)
    actual = solve_445eab21(input_grid)
    assert actual == expected


def test_445eab21_example_2():
    input_grid = ColoredGrid(values=
[[0, 4, 4, 4, 4, 4, 4, 0, 0, 0],
 [0, 4, 0, 0, 0, 0, 4, 0, 0, 0],
 [0, 4, 0, 0, 0, 0, 4, 0, 0, 0],
 [0, 4, 0, 0, 0, 0, 4, 0, 0, 0],
 [0, 4, 0, 0, 0, 0, 4, 0, 0, 0],
 [0, 4, 0, 0, 0, 0, 4, 0, 0, 0],
 [0, 4, 4, 4, 4, 4, 4, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
 [0, 0, 0, 0, 0, 0, 0, 2, 0, 2],
 [0, 0, 0, 0, 0, 0, 0, 2, 2, 2]]
    )
    expected = ColoredGrid(values=
[[4, 4], [4, 4]]
)
    actual = solve_445eab21(input_grid)
    assert actual == expected



def test_445eab21_test_case_0():
    input_grid = ColoredGrid(values=
[[3, 3, 3, 3, 3, 0, 9, 9, 9, 9],
 [3, 0, 0, 0, 3, 0, 9, 0, 0, 9],
 [3, 0, 0, 0, 3, 0, 9, 0, 0, 9],
 [3, 0, 0, 0, 3, 0, 9, 0, 0, 9],
 [3, 0, 0, 0, 3, 0, 9, 0, 0, 9],
 [3, 0, 0, 0, 3, 0, 9, 0, 0, 9],
 [3, 0, 0, 0, 3, 0, 9, 0, 0, 9],
 [3, 0, 0, 0, 3, 0, 9, 0, 0, 9],
 [3, 3, 3, 3, 3, 0, 9, 0, 0, 9],
 [0, 0, 0, 0, 0, 0, 9, 9, 9, 9]]
    )
    expected = ColoredGrid(values=
[[3, 3], [3, 3]]
    )
    actual = solve_445eab21(input_grid)
    assert actual == expected

