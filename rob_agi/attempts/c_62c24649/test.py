import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_62c24649.main import solve_62c24649


def test_62c24649_example_0():
    input_grid = ColoredGrid(values=
[[3, 3, 3], [0, 2, 2], [1, 1, 0]]
    )
    expected = ColoredGrid(values=
[[3, 3, 3, 3, 3, 3],
 [0, 2, 2, 2, 2, 0],
 [1, 1, 0, 0, 1, 1],
 [1, 1, 0, 0, 1, 1],
 [0, 2, 2, 2, 2, 0],
 [3, 3, 3, 3, 3, 3]]
)
    actual = solve_62c24649(input_grid)
    assert actual == expected


def test_62c24649_example_1():
    input_grid = ColoredGrid(values=
[[3, 3, 1], [1, 3, 0], [0, 2, 2]]
    )
    expected = ColoredGrid(values=
[[3, 3, 1, 1, 3, 3],
 [1, 3, 0, 0, 3, 1],
 [0, 2, 2, 2, 2, 0],
 [0, 2, 2, 2, 2, 0],
 [1, 3, 0, 0, 3, 1],
 [3, 3, 1, 1, 3, 3]]
)
    actual = solve_62c24649(input_grid)
    assert actual == expected


def test_62c24649_example_2():
    input_grid = ColoredGrid(values=
[[2, 1, 0], [0, 2, 3], [0, 3, 0]]
    )
    expected = ColoredGrid(values=
[[2, 1, 0, 0, 1, 2],
 [0, 2, 3, 3, 2, 0],
 [0, 3, 0, 0, 3, 0],
 [0, 3, 0, 0, 3, 0],
 [0, 2, 3, 3, 2, 0],
 [2, 1, 0, 0, 1, 2]]
)
    actual = solve_62c24649(input_grid)
    assert actual == expected



def test_62c24649_test_case_0():
    input_grid = ColoredGrid(values=
[[1, 1, 0], [0, 3, 2], [3, 3, 0]]
    )
    expected = ColoredGrid(values=
[[1, 1, 0, 0, 1, 1],
 [0, 3, 2, 2, 3, 0],
 [3, 3, 0, 0, 3, 3],
 [3, 3, 0, 0, 3, 3],
 [0, 3, 2, 2, 3, 0],
 [1, 1, 0, 0, 1, 1]]
    )
    actual = solve_62c24649(input_grid)
    assert actual == expected

