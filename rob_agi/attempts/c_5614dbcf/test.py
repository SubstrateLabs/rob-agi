import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_5614dbcf.main import solve_5614dbcf


def test_5614dbcf_example_0():
    input_grid = ColoredGrid(values=
[[3, 3, 3, 0, 0, 0, 8, 8, 8],
 [3, 3, 3, 0, 0, 0, 8, 5, 8],
 [3, 3, 3, 0, 0, 0, 8, 8, 8],
 [0, 0, 0, 7, 5, 7, 0, 0, 0],
 [0, 0, 0, 7, 7, 7, 0, 0, 0],
 [0, 0, 0, 7, 7, 7, 0, 0, 0],
 [6, 6, 6, 0, 0, 5, 9, 9, 9],
 [6, 6, 6, 0, 0, 0, 9, 9, 9],
 [6, 5, 6, 0, 5, 0, 9, 9, 5]]
    )
    expected = ColoredGrid(values=
[[3, 0, 8], [0, 7, 0], [6, 0, 9]]
)
    actual = solve_5614dbcf(input_grid)
    assert actual == expected


def test_5614dbcf_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 2, 2, 2, 0, 0, 0],
 [0, 5, 0, 2, 2, 2, 0, 0, 0],
 [0, 0, 0, 2, 2, 2, 0, 0, 0],
 [5, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 5, 0, 0, 0, 5, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 5, 0, 7, 7, 7, 0, 0, 0],
 [0, 0, 0, 7, 7, 5, 0, 0, 0],
 [0, 0, 0, 7, 7, 7, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 2, 0], [0, 0, 0], [0, 7, 0]]
)
    actual = solve_5614dbcf(input_grid)
    assert actual == expected



def test_5614dbcf_test_case_0():
    input_grid = ColoredGrid(values=
[[4, 4, 4, 0, 0, 0, 0, 5, 0],
 [5, 4, 4, 0, 0, 0, 0, 0, 0],
 [4, 4, 4, 0, 5, 0, 0, 0, 0],
 [0, 0, 0, 3, 3, 3, 0, 5, 0],
 [0, 0, 0, 3, 3, 3, 0, 0, 0],
 [0, 0, 0, 3, 3, 3, 0, 0, 0],
 [0, 0, 5, 9, 9, 9, 0, 0, 0],
 [0, 0, 0, 9, 5, 9, 0, 0, 0],
 [0, 0, 0, 9, 9, 9, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[4, 0, 0], [0, 3, 0], [0, 9, 0]]
    )
    actual = solve_5614dbcf(input_grid)
    assert actual == expected

