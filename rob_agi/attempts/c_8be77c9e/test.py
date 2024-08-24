import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_8be77c9e.main import solve_8be77c9e


def test_8be77c9e_example_0():
    input_grid = ColoredGrid(values=
[[1, 1, 0], [1, 1, 1], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 1, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 0]]
)
    actual = solve_8be77c9e(input_grid)
    assert actual == expected


def test_8be77c9e_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0], [1, 0, 1], [1, 1, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 0]]
)
    actual = solve_8be77c9e(input_grid)
    assert actual == expected


def test_8be77c9e_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0], [0, 0, 1], [0, 0, 1]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0]]
)
    actual = solve_8be77c9e(input_grid)
    assert actual == expected



def test_8be77c9e_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0], [0, 0, 1], [1, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]]
    )
    actual = solve_8be77c9e(input_grid)
    assert actual == expected

