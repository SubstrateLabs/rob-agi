import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_25ff71a9.main import solve_25ff71a9


def test_25ff71a9_example_0():
    input_grid = ColoredGrid(values=
[[1, 1, 1], [0, 0, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [1, 1, 1], [0, 0, 0]]
)
    actual = solve_25ff71a9(input_grid)
    assert actual == expected


def test_25ff71a9_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0], [1, 1, 1], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [0, 0, 0], [1, 1, 1]]
)
    actual = solve_25ff71a9(input_grid)
    assert actual == expected


def test_25ff71a9_example_2():
    input_grid = ColoredGrid(values=
[[0, 1, 0], [1, 1, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [0, 1, 0], [1, 1, 0]]
)
    actual = solve_25ff71a9(input_grid)
    assert actual == expected


def test_25ff71a9_example_3():
    input_grid = ColoredGrid(values=
[[0, 2, 2], [0, 0, 2], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [0, 2, 2], [0, 0, 2]]
)
    actual = solve_25ff71a9(input_grid)
    assert actual == expected



def test_25ff71a9_test_case_0():
    input_grid = ColoredGrid(values=
[[2, 0, 0], [2, 0, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [2, 0, 0], [2, 0, 0]]
    )
    actual = solve_25ff71a9(input_grid)
    assert actual == expected


def test_25ff71a9_test_case_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [0, 0, 0], [0, 1, 0]]
    )
    actual = solve_25ff71a9(input_grid)
    assert actual == expected

