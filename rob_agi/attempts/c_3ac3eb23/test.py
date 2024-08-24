import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_3ac3eb23.main import solve_3ac3eb23


def test_3ac3eb23_example_0():
    input_grid = ColoredGrid(values=
[[0, 2, 0, 0, 0, 8, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 2, 0, 0, 0, 8, 0, 0, 0, 0],
 [2, 0, 2, 0, 8, 0, 8, 0, 0, 0],
 [0, 2, 0, 0, 0, 8, 0, 0, 0, 0],
 [2, 0, 2, 0, 8, 0, 8, 0, 0, 0],
 [0, 2, 0, 0, 0, 8, 0, 0, 0, 0],
 [2, 0, 2, 0, 8, 0, 8, 0, 0, 0]]
)
    actual = solve_3ac3eb23(input_grid)
    assert actual == expected


def test_3ac3eb23_example_1():
    input_grid = ColoredGrid(values=
[[0, 4, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 4, 0, 0, 0, 0, 0],
 [4, 0, 4, 0, 0, 0, 0],
 [0, 4, 0, 0, 0, 0, 0],
 [4, 0, 4, 0, 0, 0, 0],
 [0, 4, 0, 0, 0, 0, 0],
 [4, 0, 4, 0, 0, 0, 0]]
)
    actual = solve_3ac3eb23(input_grid)
    assert actual == expected



def test_3ac3eb23_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 3, 0, 0, 0, 6, 0, 0, 7, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 3, 0, 0, 0, 6, 0, 0, 7, 0, 0],
 [0, 3, 0, 3, 0, 6, 0, 6, 7, 0, 7, 0],
 [0, 0, 3, 0, 0, 0, 6, 0, 0, 7, 0, 0],
 [0, 3, 0, 3, 0, 6, 0, 6, 7, 0, 7, 0],
 [0, 0, 3, 0, 0, 0, 6, 0, 0, 7, 0, 0],
 [0, 3, 0, 3, 0, 6, 0, 6, 7, 0, 7, 0]]
    )
    actual = solve_3ac3eb23(input_grid)
    assert actual == expected

