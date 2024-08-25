import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_74dd1130.main import solve_74dd1130


def test_74dd1130_example_0():
    input_grid = ColoredGrid(values=
[[2, 2, 1], [1, 5, 1], [5, 2, 2]]
    )
    expected = ColoredGrid(values=
[[2, 1, 5], [2, 5, 2], [1, 1, 2]]
)
    actual = solve_74dd1130(input_grid)
    assert actual == expected


def test_74dd1130_example_1():
    input_grid = ColoredGrid(values=
[[2, 2, 5], [6, 2, 2], [5, 5, 5]]
    )
    expected = ColoredGrid(values=
[[2, 6, 5], [2, 2, 5], [5, 2, 5]]
)
    actual = solve_74dd1130(input_grid)
    assert actual == expected


def test_74dd1130_example_2():
    input_grid = ColoredGrid(values=
[[9, 9, 5], [5, 5, 8], [5, 8, 9]]
    )
    expected = ColoredGrid(values=
[[9, 5, 5], [9, 5, 8], [5, 8, 9]]
)
    actual = solve_74dd1130(input_grid)
    assert actual == expected


def test_74dd1130_example_3():
    input_grid = ColoredGrid(values=
[[2, 6, 6], [2, 1, 1], [2, 6, 2]]
    )
    expected = ColoredGrid(values=
[[2, 2, 2], [6, 1, 6], [6, 1, 2]]
)
    actual = solve_74dd1130(input_grid)
    assert actual == expected



def test_74dd1130_test_case_0():
    input_grid = ColoredGrid(values=
[[9, 3, 4], [9, 4, 4], [9, 3, 4]]
    )
    expected = ColoredGrid(values=
[[9, 9, 9], [3, 4, 3], [4, 4, 4]]
    )
    actual = solve_74dd1130(input_grid)
    assert actual == expected

