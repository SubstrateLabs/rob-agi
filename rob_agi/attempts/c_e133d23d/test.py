import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_e133d23d.main import solve_e133d23d


def test_e133d23d_example_0():
    input_grid = ColoredGrid(values=
[[6, 0, 0, 4, 0, 0, 8], [0, 6, 0, 4, 0, 0, 8], [0, 6, 0, 4, 8, 8, 0]]
    )
    expected = ColoredGrid(values=
[[2, 0, 2], [0, 2, 2], [2, 2, 0]]
)
    actual = solve_e133d23d(input_grid)
    assert actual == expected


def test_e133d23d_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 6, 4, 8, 8, 0], [0, 6, 0, 4, 0, 8, 8], [0, 6, 6, 4, 8, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 2, 2], [0, 2, 2], [2, 2, 2]]
)
    actual = solve_e133d23d(input_grid)
    assert actual == expected


def test_e133d23d_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 6, 4, 8, 0, 8], [6, 0, 6, 4, 0, 0, 0], [0, 6, 6, 4, 8, 0, 8]]
    )
    expected = ColoredGrid(values=
[[2, 0, 2], [2, 0, 2], [2, 2, 2]]
)
    actual = solve_e133d23d(input_grid)
    assert actual == expected


def test_e133d23d_example_3():
    input_grid = ColoredGrid(values=
[[6, 0, 6, 4, 0, 0, 0], [6, 6, 0, 4, 8, 0, 8], [6, 6, 6, 4, 0, 8, 0]]
    )
    expected = ColoredGrid(values=
[[2, 0, 2], [2, 2, 2], [2, 2, 2]]
)
    actual = solve_e133d23d(input_grid)
    assert actual == expected


def test_e133d23d_example_4():
    input_grid = ColoredGrid(values=
[[0, 0, 6, 4, 8, 0, 8], [0, 6, 0, 4, 0, 8, 0], [0, 0, 0, 4, 8, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 0, 2], [0, 2, 0], [2, 0, 0]]
)
    actual = solve_e133d23d(input_grid)
    assert actual == expected



def test_e133d23d_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 6, 6, 4, 0, 0, 8], [0, 6, 0, 4, 8, 8, 8], [6, 0, 6, 4, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 2, 2], [2, 2, 2], [2, 0, 2]]
    )
    actual = solve_e133d23d(input_grid)
    assert actual == expected

