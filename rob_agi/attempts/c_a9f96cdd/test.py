import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_a9f96cdd.main import solve_a9f96cdd


def test_a9f96cdd_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[3, 0, 6, 0, 0], [0, 0, 0, 0, 0], [8, 0, 7, 0, 0]]
)
    actual = solve_a9f96cdd(input_grid)
    assert actual == expected


def test_a9f96cdd_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 2]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0], [0, 0, 0, 3, 0], [0, 0, 0, 0, 0]]
)
    actual = solve_a9f96cdd(input_grid)
    assert actual == expected


def test_a9f96cdd_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 2, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0], [0, 8, 0, 7, 0], [0, 0, 0, 0, 0]]
)
    actual = solve_a9f96cdd(input_grid)
    assert actual == expected


def test_a9f96cdd_example_3():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 3, 0, 6], [0, 0, 0, 0, 0], [0, 0, 8, 0, 7]]
)
    actual = solve_a9f96cdd(input_grid)
    assert actual == expected



def test_a9f96cdd_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0], [0, 0, 0, 0, 2], [0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 3, 0], [0, 0, 0, 0, 0], [0, 0, 0, 8, 0]]
    )
    actual = solve_a9f96cdd(input_grid)
    assert actual == expected

