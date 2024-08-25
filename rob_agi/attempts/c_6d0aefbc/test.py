import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_6d0aefbc.main import solve_6d0aefbc


def test_6d0aefbc_example_0():
    input_grid = ColoredGrid(values=
[[6, 6, 6], [1, 6, 1], [8, 8, 6]]
    )
    expected = ColoredGrid(values=
[[6, 6, 6, 6, 6, 6], [1, 6, 1, 1, 6, 1], [8, 8, 6, 6, 8, 8]]
)
    actual = solve_6d0aefbc(input_grid)
    assert actual == expected


def test_6d0aefbc_example_1():
    input_grid = ColoredGrid(values=
[[6, 8, 1], [6, 1, 1], [1, 1, 6]]
    )
    expected = ColoredGrid(values=
[[6, 8, 1, 1, 8, 6], [6, 1, 1, 1, 1, 6], [1, 1, 6, 6, 1, 1]]
)
    actual = solve_6d0aefbc(input_grid)
    assert actual == expected


def test_6d0aefbc_example_2():
    input_grid = ColoredGrid(values=
[[1, 1, 1], [8, 1, 6], [6, 8, 8]]
    )
    expected = ColoredGrid(values=
[[1, 1, 1, 1, 1, 1], [8, 1, 6, 6, 1, 8], [6, 8, 8, 8, 8, 6]]
)
    actual = solve_6d0aefbc(input_grid)
    assert actual == expected


def test_6d0aefbc_example_3():
    input_grid = ColoredGrid(values=
[[1, 1, 1], [1, 6, 6], [6, 6, 6]]
    )
    expected = ColoredGrid(values=
[[1, 1, 1, 1, 1, 1], [1, 6, 6, 6, 6, 1], [6, 6, 6, 6, 6, 6]]
)
    actual = solve_6d0aefbc(input_grid)
    assert actual == expected



def test_6d0aefbc_test_case_0():
    input_grid = ColoredGrid(values=
[[6, 8, 6], [8, 6, 8], [1, 6, 1]]
    )
    expected = ColoredGrid(values=
[[6, 8, 6, 6, 8, 6], [8, 6, 8, 8, 6, 8], [1, 6, 1, 1, 6, 1]]
    )
    actual = solve_6d0aefbc(input_grid)
    assert actual == expected

