import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_be94b721.main import solve_be94b721


def test_be94b721_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 2, 2, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0],
 [0, 0, 2, 0, 0, 0, 3, 0, 0, 1, 0, 0, 0],
 [0, 0, 2, 2, 0, 0, 0, 0, 1, 1, 0, 0, 0],
 [0, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 2, 0], [0, 2, 0], [0, 2, 2], [2, 2, 2]]
)
    actual = solve_be94b721(input_grid)
    assert actual == expected


def test_be94b721_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 6, 6, 0],
 [0, 3, 0, 0, 4, 4, 0, 0, 6, 0],
 [3, 3, 3, 0, 4, 4, 0, 0, 0, 0],
 [0, 3, 0, 0, 4, 4, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[4, 4], [4, 4], [4, 4]]
)
    actual = solve_be94b721(input_grid)
    assert actual == expected


def test_be94b721_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 8, 8, 8, 0, 0, 0, 0, 7, 7, 0],
 [0, 0, 8, 0, 0, 0, 2, 0, 0, 7, 0],
 [0, 8, 8, 0, 0, 2, 2, 0, 0, 7, 0],
 [0, 8, 8, 0, 0, 0, 2, 0, 0, 7, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 8, 8], [0, 8, 0], [8, 8, 0], [8, 8, 0]]
)
    actual = solve_be94b721(input_grid)
    assert actual == expected


def test_be94b721_example_3():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 7, 0, 0, 2, 2, 2],
 [0, 0, 0, 7, 7, 0, 0, 2, 0],
 [0, 0, 0, 0, 7, 0, 2, 2, 2],
 [8, 8, 8, 0, 0, 0, 0, 0, 0],
 [0, 8, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 2, 2], [0, 2, 0], [2, 2, 2]]
)
    actual = solve_be94b721(input_grid)
    assert actual == expected



def test_be94b721_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [4, 0, 0, 0, 0, 0, 0, 0, 0],
 [4, 4, 0, 3, 3, 3, 0, 0, 0],
 [0, 4, 0, 3, 3, 3, 0, 0, 0],
 [0, 0, 0, 3, 0, 3, 0, 0, 0],
 [0, 0, 0, 3, 0, 3, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 6, 6],
 [0, 5, 5, 5, 0, 0, 6, 6, 6],
 [0, 5, 5, 0, 0, 0, 6, 6, 0]]
    )
    expected = ColoredGrid(values=
[[3, 3, 3], [3, 3, 3], [3, 0, 3], [3, 0, 3]]
    )
    actual = solve_be94b721(input_grid)
    assert actual == expected

