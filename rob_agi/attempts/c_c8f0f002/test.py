import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_c8f0f002.main import solve_c8f0f002


def test_c8f0f002_example_0():
    input_grid = ColoredGrid(values=
[[1, 8, 8, 7, 7, 8], [1, 1, 7, 7, 1, 8], [7, 1, 1, 7, 7, 8]]
    )
    expected = ColoredGrid(values=
[[1, 8, 8, 5, 5, 8], [1, 1, 5, 5, 1, 8], [5, 1, 1, 5, 5, 8]]
)
    actual = solve_c8f0f002(input_grid)
    assert actual == expected


def test_c8f0f002_example_1():
    input_grid = ColoredGrid(values=
[[7, 7, 7, 1], [1, 8, 1, 7], [7, 1, 1, 7]]
    )
    expected = ColoredGrid(values=
[[5, 5, 5, 1], [1, 8, 1, 5], [5, 1, 1, 5]]
)
    actual = solve_c8f0f002(input_grid)
    assert actual == expected


def test_c8f0f002_example_2():
    input_grid = ColoredGrid(values=
[[1, 8, 1, 7, 1], [7, 8, 8, 1, 1], [7, 1, 8, 8, 7]]
    )
    expected = ColoredGrid(values=
[[1, 8, 1, 5, 1], [5, 8, 8, 1, 1], [5, 1, 8, 8, 5]]
)
    actual = solve_c8f0f002(input_grid)
    assert actual == expected



def test_c8f0f002_test_case_0():
    input_grid = ColoredGrid(values=
[[1, 7, 7, 1, 7], [8, 1, 7, 7, 7], [8, 7, 1, 7, 8]]
    )
    expected = ColoredGrid(values=
[[1, 5, 5, 1, 5], [8, 1, 5, 5, 5], [8, 5, 1, 5, 8]]
    )
    actual = solve_c8f0f002(input_grid)
    assert actual == expected

