import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_bdad9b1f.main import solve_bdad9b1f


def test_bdad9b1f_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 8, 0],
 [0, 0, 0, 0, 8, 0],
 [2, 2, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 8, 0],
 [0, 0, 0, 0, 8, 0],
 [2, 2, 2, 2, 4, 2],
 [0, 0, 0, 0, 8, 0],
 [0, 0, 0, 0, 8, 0],
 [0, 0, 0, 0, 8, 0]]
)
    actual = solve_bdad9b1f(input_grid)
    assert actual == expected


def test_bdad9b1f_example_1():
    input_grid = ColoredGrid(values=
[[0, 8, 0, 0, 0, 0],
 [0, 8, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 2, 2],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 8, 0, 0, 0, 0],
 [0, 8, 0, 0, 0, 0],
 [0, 8, 0, 0, 0, 0],
 [2, 4, 2, 2, 2, 2],
 [0, 8, 0, 0, 0, 0],
 [0, 8, 0, 0, 0, 0]]
)
    actual = solve_bdad9b1f(input_grid)
    assert actual == expected



def test_bdad9b1f_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 8, 0, 0],
 [0, 0, 0, 8, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [2, 2, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 8, 0, 0],
 [0, 0, 0, 8, 0, 0],
 [0, 0, 0, 8, 0, 0],
 [0, 0, 0, 8, 0, 0],
 [2, 2, 2, 4, 2, 2],
 [0, 0, 0, 8, 0, 0]]
    )
    actual = solve_bdad9b1f(input_grid)
    assert actual == expected

