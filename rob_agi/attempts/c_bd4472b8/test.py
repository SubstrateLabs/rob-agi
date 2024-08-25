import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_bd4472b8.main import solve_bd4472b8


def test_bd4472b8_example_0():
    input_grid = ColoredGrid(values=
[[2, 1, 4],
 [5, 5, 5],
 [0, 0, 0],
 [0, 0, 0],
 [0, 0, 0],
 [0, 0, 0],
 [0, 0, 0],
 [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 1, 4],
 [5, 5, 5],
 [2, 2, 2],
 [1, 1, 1],
 [4, 4, 4],
 [2, 2, 2],
 [1, 1, 1],
 [4, 4, 4]]
)
    actual = solve_bd4472b8(input_grid)
    assert actual == expected


def test_bd4472b8_example_1():
    input_grid = ColoredGrid(values=
[[3, 2, 1, 4],
 [5, 5, 5, 5],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[3, 2, 1, 4],
 [5, 5, 5, 5],
 [3, 3, 3, 3],
 [2, 2, 2, 2],
 [1, 1, 1, 1],
 [4, 4, 4, 4],
 [3, 3, 3, 3],
 [2, 2, 2, 2],
 [1, 1, 1, 1],
 [4, 4, 4, 4]]
)
    actual = solve_bd4472b8(input_grid)
    assert actual == expected


def test_bd4472b8_example_2():
    input_grid = ColoredGrid(values=
[[8, 3], [5, 5], [0, 0], [0, 0], [0, 0], [0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 3], [5, 5], [8, 8], [3, 3], [8, 8], [3, 3]]
)
    actual = solve_bd4472b8(input_grid)
    assert actual == expected



def test_bd4472b8_test_case_0():
    input_grid = ColoredGrid(values=
[[1, 2, 3, 4, 8],
 [5, 5, 5, 5, 5],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 2, 3, 4, 8],
 [5, 5, 5, 5, 5],
 [1, 1, 1, 1, 1],
 [2, 2, 2, 2, 2],
 [3, 3, 3, 3, 3],
 [4, 4, 4, 4, 4],
 [8, 8, 8, 8, 8],
 [1, 1, 1, 1, 1],
 [2, 2, 2, 2, 2],
 [3, 3, 3, 3, 3],
 [4, 4, 4, 4, 4],
 [8, 8, 8, 8, 8]]
    )
    actual = solve_bd4472b8(input_grid)
    assert actual == expected

