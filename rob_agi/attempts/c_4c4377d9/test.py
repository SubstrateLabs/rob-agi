import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_4c4377d9.main import solve_4c4377d9


def test_4c4377d9_example_0():
    input_grid = ColoredGrid(values=
[[9, 9, 5, 9], [5, 5, 9, 9], [9, 5, 9, 9]]
    )
    expected = ColoredGrid(values=
[[9, 5, 9, 9],
 [5, 5, 9, 9],
 [9, 9, 5, 9],
 [9, 9, 5, 9],
 [5, 5, 9, 9],
 [9, 5, 9, 9]]
)
    actual = solve_4c4377d9(input_grid)
    assert actual == expected


def test_4c4377d9_example_1():
    input_grid = ColoredGrid(values=
[[4, 1, 1, 4], [1, 1, 1, 1], [4, 4, 4, 1]]
    )
    expected = ColoredGrid(values=
[[4, 4, 4, 1],
 [1, 1, 1, 1],
 [4, 1, 1, 4],
 [4, 1, 1, 4],
 [1, 1, 1, 1],
 [4, 4, 4, 1]]
)
    actual = solve_4c4377d9(input_grid)
    assert actual == expected


def test_4c4377d9_example_2():
    input_grid = ColoredGrid(values=
[[9, 4, 9, 4], [9, 9, 4, 4], [4, 4, 4, 4]]
    )
    expected = ColoredGrid(values=
[[4, 4, 4, 4],
 [9, 9, 4, 4],
 [9, 4, 9, 4],
 [9, 4, 9, 4],
 [9, 9, 4, 4],
 [4, 4, 4, 4]]
)
    actual = solve_4c4377d9(input_grid)
    assert actual == expected


def test_4c4377d9_example_3():
    input_grid = ColoredGrid(values=
[[3, 3, 5, 5], [3, 5, 5, 3], [5, 5, 3, 3]]
    )
    expected = ColoredGrid(values=
[[5, 5, 3, 3],
 [3, 5, 5, 3],
 [3, 3, 5, 5],
 [3, 3, 5, 5],
 [3, 5, 5, 3],
 [5, 5, 3, 3]]
)
    actual = solve_4c4377d9(input_grid)
    assert actual == expected



def test_4c4377d9_test_case_0():
    input_grid = ColoredGrid(values=
[[4, 4, 9, 9], [4, 4, 4, 4], [4, 4, 9, 9]]
    )
    expected = ColoredGrid(values=
[[4, 4, 9, 9],
 [4, 4, 4, 4],
 [4, 4, 9, 9],
 [4, 4, 9, 9],
 [4, 4, 4, 4],
 [4, 4, 9, 9]]
    )
    actual = solve_4c4377d9(input_grid)
    assert actual == expected

