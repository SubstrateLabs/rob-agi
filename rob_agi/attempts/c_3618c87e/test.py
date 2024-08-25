import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_3618c87e.main import solve_3618c87e


def test_3618c87e_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 1, 0, 0],
 [0, 0, 5, 0, 0],
 [5, 5, 5, 5, 5]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 5, 0, 0],
 [5, 5, 1, 5, 5]]
)
    actual = solve_3618c87e(input_grid)
    assert actual == expected


def test_3618c87e_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 1, 0, 1, 0],
 [0, 5, 0, 5, 0],
 [5, 5, 5, 5, 5]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 5, 0, 5, 0],
 [5, 1, 5, 1, 5]]
)
    actual = solve_3618c87e(input_grid)
    assert actual == expected


def test_3618c87e_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 1, 0, 0, 1],
 [0, 5, 0, 0, 5],
 [5, 5, 5, 5, 5]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 5, 0, 0, 5],
 [5, 1, 5, 5, 1]]
)
    actual = solve_3618c87e(input_grid)
    assert actual == expected



def test_3618c87e_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 1, 0, 1],
 [0, 0, 5, 0, 5],
 [5, 5, 5, 5, 5]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 5, 0, 5],
 [5, 5, 1, 5, 1]]
    )
    actual = solve_3618c87e(input_grid)
    assert actual == expected

