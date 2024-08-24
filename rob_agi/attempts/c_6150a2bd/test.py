import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_6150a2bd.main import solve_6150a2bd


def test_6150a2bd_example_0():
    input_grid = ColoredGrid(values=
[[3, 3, 8], [3, 7, 0], [5, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 5], [0, 7, 3], [8, 3, 3]]
)
    actual = solve_6150a2bd(input_grid)
    assert actual == expected


def test_6150a2bd_example_1():
    input_grid = ColoredGrid(values=
[[5, 5, 2], [1, 0, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [0, 0, 1], [2, 5, 5]]
)
    actual = solve_6150a2bd(input_grid)
    assert actual == expected



def test_6150a2bd_test_case_0():
    input_grid = ColoredGrid(values=
[[6, 3, 5], [6, 8, 0], [4, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 4], [0, 8, 6], [5, 3, 6]]
    )
    actual = solve_6150a2bd(input_grid)
    assert actual == expected

