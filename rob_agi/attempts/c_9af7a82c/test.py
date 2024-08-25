import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_9af7a82c.main import solve_9af7a82c


def test_9af7a82c_example_0():
    input_grid = ColoredGrid(values=
[[2, 2, 1], [2, 3, 1], [1, 1, 1]]
    )
    expected = ColoredGrid(values=
[[1, 2, 3], [1, 2, 0], [1, 2, 0], [1, 0, 0], [1, 0, 0]]
)
    actual = solve_9af7a82c(input_grid)
    assert actual == expected


def test_9af7a82c_example_1():
    input_grid = ColoredGrid(values=
[[3, 1, 1, 4], [2, 2, 2, 4], [4, 4, 4, 4]]
    )
    expected = ColoredGrid(values=
[[4, 2, 1, 3],
 [4, 2, 1, 0],
 [4, 2, 0, 0],
 [4, 0, 0, 0],
 [4, 0, 0, 0],
 [4, 0, 0, 0]]
)
    actual = solve_9af7a82c(input_grid)
    assert actual == expected


def test_9af7a82c_example_2():
    input_grid = ColoredGrid(values=
[[8, 8, 2], [3, 8, 8], [3, 3, 4], [3, 3, 4]]
    )
    expected = ColoredGrid(values=
[[3, 8, 4, 2], [3, 8, 4, 0], [3, 8, 0, 0], [3, 8, 0, 0], [3, 0, 0, 0]]
)
    actual = solve_9af7a82c(input_grid)
    assert actual == expected


def test_9af7a82c_example_3():
    input_grid = ColoredGrid(values=
[[1, 1, 1], [2, 2, 1], [2, 8, 1], [2, 8, 1]]
    )
    expected = ColoredGrid(values=
[[1, 2, 8], [1, 2, 8], [1, 2, 0], [1, 2, 0], [1, 0, 0], [1, 0, 0]]
)
    actual = solve_9af7a82c(input_grid)
    assert actual == expected



def test_9af7a82c_test_case_0():
    input_grid = ColoredGrid(values=
[[8, 8, 2, 2], [1, 8, 8, 2], [1, 3, 3, 4], [1, 1, 1, 1]]
    )
    expected = ColoredGrid(values=
[[1, 8, 2, 3, 4],
 [1, 8, 2, 3, 0],
 [1, 8, 2, 0, 0],
 [1, 8, 0, 0, 0],
 [1, 0, 0, 0, 0],
 [1, 0, 0, 0, 0]]
    )
    actual = solve_9af7a82c(input_grid)
    assert actual == expected

