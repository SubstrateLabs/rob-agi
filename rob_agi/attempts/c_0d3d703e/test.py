import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_0d3d703e.main import solve_0d3d703e


def test_0d3d703e_example_0():
    input_grid = ColoredGrid(values=
[[3, 1, 2], [3, 1, 2], [3, 1, 2]]
    )
    expected = ColoredGrid(values=
[[4, 5, 6], [4, 5, 6], [4, 5, 6]]
)
    actual = solve_0d3d703e(input_grid)
    assert actual == expected


def test_0d3d703e_example_1():
    input_grid = ColoredGrid(values=
[[2, 3, 8], [2, 3, 8], [2, 3, 8]]
    )
    expected = ColoredGrid(values=
[[6, 4, 9], [6, 4, 9], [6, 4, 9]]
)
    actual = solve_0d3d703e(input_grid)
    assert actual == expected


def test_0d3d703e_example_2():
    input_grid = ColoredGrid(values=
[[5, 8, 6], [5, 8, 6], [5, 8, 6]]
    )
    expected = ColoredGrid(values=
[[1, 9, 2], [1, 9, 2], [1, 9, 2]]
)
    actual = solve_0d3d703e(input_grid)
    assert actual == expected


def test_0d3d703e_example_3():
    input_grid = ColoredGrid(values=
[[9, 4, 2], [9, 4, 2], [9, 4, 2]]
    )
    expected = ColoredGrid(values=
[[8, 3, 6], [8, 3, 6], [8, 3, 6]]
)
    actual = solve_0d3d703e(input_grid)
    assert actual == expected



def test_0d3d703e_test_case_0():
    input_grid = ColoredGrid(values=
[[8, 1, 3], [8, 1, 3], [8, 1, 3]]
    )
    expected = ColoredGrid(values=
[[9, 5, 4], [9, 5, 4], [9, 5, 4]]
    )
    actual = solve_0d3d703e(input_grid)
    assert actual == expected

