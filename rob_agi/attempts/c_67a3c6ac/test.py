import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_67a3c6ac.main import solve_67a3c6ac


def test_67a3c6ac_example_0():
    input_grid = ColoredGrid(values=
[[6, 6, 6, 2], [6, 1, 6, 2], [7, 2, 7, 2], [1, 7, 2, 2]]
    )
    expected = ColoredGrid(values=
[[2, 6, 6, 6], [2, 6, 1, 6], [2, 7, 2, 7], [2, 2, 7, 1]]
)
    actual = solve_67a3c6ac(input_grid)
    assert actual == expected


def test_67a3c6ac_example_1():
    input_grid = ColoredGrid(values=
[[7, 7, 7, 6, 6, 6, 2],
 [6, 7, 1, 1, 7, 7, 1],
 [7, 7, 2, 1, 2, 6, 6],
 [2, 2, 7, 7, 7, 2, 2],
 [7, 2, 7, 1, 2, 7, 2],
 [6, 6, 6, 2, 2, 1, 1],
 [6, 2, 6, 6, 6, 6, 6]]
    )
    expected = ColoredGrid(values=
[[2, 6, 6, 6, 7, 7, 7],
 [1, 7, 7, 1, 1, 7, 6],
 [6, 6, 2, 1, 2, 7, 7],
 [2, 2, 7, 7, 7, 2, 2],
 [2, 7, 2, 1, 7, 2, 7],
 [1, 1, 2, 2, 6, 6, 6],
 [6, 6, 6, 6, 6, 2, 6]]
)
    actual = solve_67a3c6ac(input_grid)
    assert actual == expected


def test_67a3c6ac_example_2():
    input_grid = ColoredGrid(values=
[[1, 2, 7, 1, 1, 1],
 [2, 1, 7, 7, 2, 6],
 [2, 1, 2, 6, 2, 1],
 [1, 2, 1, 7, 6, 2],
 [2, 7, 1, 2, 7, 1],
 [2, 1, 6, 2, 7, 7]]
    )
    expected = ColoredGrid(values=
[[1, 1, 1, 7, 2, 1],
 [6, 2, 7, 7, 1, 2],
 [1, 2, 6, 2, 1, 2],
 [2, 6, 7, 1, 2, 1],
 [1, 7, 2, 1, 7, 2],
 [7, 7, 2, 6, 1, 2]]
)
    actual = solve_67a3c6ac(input_grid)
    assert actual == expected



def test_67a3c6ac_test_case_0():
    input_grid = ColoredGrid(values=
[[7, 6, 1], [6, 7, 6], [6, 2, 2]]
    )
    expected = ColoredGrid(values=
[[1, 6, 7], [6, 7, 6], [2, 2, 6]]
    )
    actual = solve_67a3c6ac(input_grid)
    assert actual == expected

