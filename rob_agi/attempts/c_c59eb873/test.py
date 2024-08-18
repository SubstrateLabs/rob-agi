
import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_c59eb873.main import solve_c59eb873


def test_c59eb873_example_0():
    input_grid = ColoredGrid(values=
[[0, 5, 1], [5, 5, 5], [2, 5, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 5, 5, 1, 1],
 [0, 0, 5, 5, 1, 1],
 [5, 5, 5, 5, 5, 5],
 [5, 5, 5, 5, 5, 5],
 [2, 2, 5, 5, 0, 0],
 [2, 2, 5, 5, 0, 0]]
)
    actual = solve_c59eb873(input_grid)
    assert actual == expected


def test_c59eb873_example_1():
    input_grid = ColoredGrid(values=
[[2, 1], [3, 1]]
    )
    expected = ColoredGrid(values=
[[2, 2, 1, 1], [2, 2, 1, 1], [3, 3, 1, 1], [3, 3, 1, 1]]
)
    actual = solve_c59eb873(input_grid)
    assert actual == expected


def test_c59eb873_example_2():
    input_grid = ColoredGrid(values=
[[2, 0, 3, 0], [2, 1, 3, 0], [0, 0, 3, 3], [0, 0, 3, 5]]
    )
    expected = ColoredGrid(values=
[[2, 2, 0, 0, 3, 3, 0, 0],
 [2, 2, 0, 0, 3, 3, 0, 0],
 [2, 2, 1, 1, 3, 3, 0, 0],
 [2, 2, 1, 1, 3, 3, 0, 0],
 [0, 0, 0, 0, 3, 3, 3, 3],
 [0, 0, 0, 0, 3, 3, 3, 3],
 [0, 0, 0, 0, 3, 3, 5, 5],
 [0, 0, 0, 0, 3, 3, 5, 5]]
)
    actual = solve_c59eb873(input_grid)
    assert actual == expected



def test_c59eb873_test_case_0():
    input_grid = ColoredGrid(values=
[[2, 0, 0, 7, 8],
 [2, 1, 1, 0, 0],
 [0, 5, 6, 6, 0],
 [3, 5, 6, 0, 0],
 [0, 5, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 2, 0, 0, 0, 0, 7, 7, 8, 8],
 [2, 2, 0, 0, 0, 0, 7, 7, 8, 8],
 [2, 2, 1, 1, 1, 1, 0, 0, 0, 0],
 [2, 2, 1, 1, 1, 1, 0, 0, 0, 0],
 [0, 0, 5, 5, 6, 6, 6, 6, 0, 0],
 [0, 0, 5, 5, 6, 6, 6, 6, 0, 0],
 [3, 3, 5, 5, 6, 6, 0, 0, 0, 0],
 [3, 3, 5, 5, 6, 6, 0, 0, 0, 0],
 [0, 0, 5, 5, 0, 0, 0, 0, 0, 0],
 [0, 0, 5, 5, 0, 0, 0, 0, 0, 0]]
    )
    actual = solve_c59eb873(input_grid)
    assert actual == expected

