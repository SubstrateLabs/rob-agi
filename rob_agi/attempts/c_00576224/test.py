import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_00576224.main import solve_00576224


def test_00576224_example_0():
    input_grid = ColoredGrid(values=
[[8, 6], [6, 4]]
    )
    expected = ColoredGrid(values=
[[8, 6, 8, 6, 8, 6],
 [6, 4, 6, 4, 6, 4],
 [6, 8, 6, 8, 6, 8],
 [4, 6, 4, 6, 4, 6],
 [8, 6, 8, 6, 8, 6],
 [6, 4, 6, 4, 6, 4]]
)
    actual = solve_00576224(input_grid)
    assert actual == expected


def test_00576224_example_1():
    input_grid = ColoredGrid(values=
[[7, 9], [4, 3]]
    )
    expected = ColoredGrid(values=
[[7, 9, 7, 9, 7, 9],
 [4, 3, 4, 3, 4, 3],
 [9, 7, 9, 7, 9, 7],
 [3, 4, 3, 4, 3, 4],
 [7, 9, 7, 9, 7, 9],
 [4, 3, 4, 3, 4, 3]]
)
    actual = solve_00576224(input_grid)
    assert actual == expected



def test_00576224_test_case_0():
    input_grid = ColoredGrid(values=
[[3, 2], [7, 8]]
    )
    expected = ColoredGrid(values=
[[3, 2, 3, 2, 3, 2],
 [7, 8, 7, 8, 7, 8],
 [2, 3, 2, 3, 2, 3],
 [8, 7, 8, 7, 8, 7],
 [3, 2, 3, 2, 3, 2],
 [7, 8, 7, 8, 7, 8]]
    )
    actual = solve_00576224(input_grid)
    assert actual == expected

