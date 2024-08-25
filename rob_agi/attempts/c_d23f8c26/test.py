import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_d23f8c26.main import solve_d23f8c26


def test_d23f8c26_example_0():
    input_grid = ColoredGrid(values=
[[6, 4, 0], [0, 3, 9], [1, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 4, 0], [0, 3, 0], [0, 0, 0]]
)
    actual = solve_d23f8c26(input_grid)
    assert actual == expected


def test_d23f8c26_example_1():
    input_grid = ColoredGrid(values=
[[8, 0, 3, 0, 0],
 [8, 6, 5, 6, 0],
 [3, 6, 3, 0, 0],
 [0, 0, 0, 5, 9],
 [5, 0, 9, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 3, 0, 0],
 [0, 0, 5, 0, 0],
 [0, 0, 3, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 9, 0, 0]]
)
    actual = solve_d23f8c26(input_grid)
    assert actual == expected


def test_d23f8c26_example_2():
    input_grid = ColoredGrid(values=
[[3, 0, 4, 0, 0],
 [3, 0, 4, 7, 0],
 [0, 6, 0, 0, 7],
 [0, 0, 8, 0, 0],
 [0, 8, 0, 2, 2]]
    )
    expected = ColoredGrid(values=
[[0, 0, 4, 0, 0],
 [0, 0, 4, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 8, 0, 0],
 [0, 0, 0, 0, 0]]
)
    actual = solve_d23f8c26(input_grid)
    assert actual == expected



def test_d23f8c26_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 3, 0, 0, 0, 7],
 [8, 1, 0, 8, 0, 0, 0],
 [0, 0, 3, 0, 8, 0, 3],
 [0, 7, 0, 1, 0, 7, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [1, 0, 8, 6, 0, 0, 0],
 [0, 8, 0, 6, 0, 1, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 8, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 6, 0, 0, 0],
 [0, 0, 0, 6, 0, 0, 0]]
    )
    actual = solve_d23f8c26(input_grid)
    assert actual == expected

