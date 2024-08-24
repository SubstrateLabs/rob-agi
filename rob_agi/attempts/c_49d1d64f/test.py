import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_49d1d64f.main import solve_49d1d64f


def test_49d1d64f_example_0():
    input_grid = ColoredGrid(values=
[[1, 2], [3, 8]]
    )
    expected = ColoredGrid(values=
[[0, 1, 2, 0], [1, 1, 2, 2], [3, 3, 8, 8], [0, 3, 8, 0]]
)
    actual = solve_49d1d64f(input_grid)
    assert actual == expected


def test_49d1d64f_example_1():
    input_grid = ColoredGrid(values=
[[1, 8, 4], [8, 3, 8]]
    )
    expected = ColoredGrid(values=
[[0, 1, 8, 4, 0], [1, 1, 8, 4, 4], [8, 8, 3, 8, 8], [0, 8, 3, 8, 0]]
)
    actual = solve_49d1d64f(input_grid)
    assert actual == expected


def test_49d1d64f_example_2():
    input_grid = ColoredGrid(values=
[[2, 1, 4], [8, 0, 2], [3, 2, 8]]
    )
    expected = ColoredGrid(values=
[[0, 2, 1, 4, 0],
 [2, 2, 1, 4, 4],
 [8, 8, 0, 2, 2],
 [3, 3, 2, 8, 8],
 [0, 3, 2, 8, 0]]
)
    actual = solve_49d1d64f(input_grid)
    assert actual == expected



def test_49d1d64f_test_case_0():
    input_grid = ColoredGrid(values=
[[2, 8], [1, 4], [3, 4]]
    )
    expected = ColoredGrid(values=
[[0, 2, 8, 0], [2, 2, 8, 8], [1, 1, 4, 4], [3, 3, 4, 4], [0, 3, 4, 0]]
    )
    actual = solve_49d1d64f(input_grid)
    assert actual == expected

