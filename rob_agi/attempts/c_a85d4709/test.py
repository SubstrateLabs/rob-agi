import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_a85d4709.main import solve_a85d4709


def test_a85d4709_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 5], [0, 5, 0], [5, 0, 0]]
    )
    expected = ColoredGrid(values=
[[3, 3, 3], [4, 4, 4], [2, 2, 2]]
)
    actual = solve_a85d4709(input_grid)
    assert actual == expected


def test_a85d4709_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 5], [0, 0, 5], [0, 0, 5]]
    )
    expected = ColoredGrid(values=
[[3, 3, 3], [3, 3, 3], [3, 3, 3]]
)
    actual = solve_a85d4709(input_grid)
    assert actual == expected


def test_a85d4709_example_2():
    input_grid = ColoredGrid(values=
[[5, 0, 0], [0, 5, 0], [5, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 2, 2], [4, 4, 4], [2, 2, 2]]
)
    actual = solve_a85d4709(input_grid)
    assert actual == expected


def test_a85d4709_example_3():
    input_grid = ColoredGrid(values=
[[0, 5, 0], [0, 0, 5], [0, 5, 0]]
    )
    expected = ColoredGrid(values=
[[4, 4, 4], [3, 3, 3], [4, 4, 4]]
)
    actual = solve_a85d4709(input_grid)
    assert actual == expected



def test_a85d4709_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 5], [5, 0, 0], [0, 5, 0]]
    )
    expected = ColoredGrid(values=
[[3, 3, 3], [2, 2, 2], [4, 4, 4]]
    )
    actual = solve_a85d4709(input_grid)
    assert actual == expected

