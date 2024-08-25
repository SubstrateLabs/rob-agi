import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_c48954c1.main import solve_c48954c1


def test_c48954c1_example_0():
    input_grid = ColoredGrid(values=
[[7, 6, 7], [2, 7, 6], [1, 2, 7]]
    )
    expected = ColoredGrid(values=
[[7, 2, 1, 1, 2, 7, 7, 2, 1],
 [6, 7, 2, 2, 7, 6, 6, 7, 2],
 [7, 6, 7, 7, 6, 7, 7, 6, 7],
 [7, 6, 7, 7, 6, 7, 7, 6, 7],
 [6, 7, 2, 2, 7, 6, 6, 7, 2],
 [7, 2, 1, 1, 2, 7, 7, 2, 1],
 [7, 2, 1, 1, 2, 7, 7, 2, 1],
 [6, 7, 2, 2, 7, 6, 6, 7, 2],
 [7, 6, 7, 7, 6, 7, 7, 6, 7]]
)
    actual = solve_c48954c1(input_grid)
    assert actual == expected


def test_c48954c1_example_1():
    input_grid = ColoredGrid(values=
[[6, 1, 7], [1, 6, 7], [4, 7, 4]]
    )
    expected = ColoredGrid(values=
[[4, 7, 4, 4, 7, 4, 4, 7, 4],
 [7, 6, 1, 1, 6, 7, 7, 6, 1],
 [7, 1, 6, 6, 1, 7, 7, 1, 6],
 [7, 1, 6, 6, 1, 7, 7, 1, 6],
 [7, 6, 1, 1, 6, 7, 7, 6, 1],
 [4, 7, 4, 4, 7, 4, 4, 7, 4],
 [4, 7, 4, 4, 7, 4, 4, 7, 4],
 [7, 6, 1, 1, 6, 7, 7, 6, 1],
 [7, 1, 6, 6, 1, 7, 7, 1, 6]]
)
    actual = solve_c48954c1(input_grid)
    assert actual == expected


def test_c48954c1_example_2():
    input_grid = ColoredGrid(values=
[[1, 9, 4], [9, 1, 6], [6, 9, 4]]
    )
    expected = ColoredGrid(values=
[[4, 9, 6, 6, 9, 4, 4, 9, 6],
 [6, 1, 9, 9, 1, 6, 6, 1, 9],
 [4, 9, 1, 1, 9, 4, 4, 9, 1],
 [4, 9, 1, 1, 9, 4, 4, 9, 1],
 [6, 1, 9, 9, 1, 6, 6, 1, 9],
 [4, 9, 6, 6, 9, 4, 4, 9, 6],
 [4, 9, 6, 6, 9, 4, 4, 9, 6],
 [6, 1, 9, 9, 1, 6, 6, 1, 9],
 [4, 9, 1, 1, 9, 4, 4, 9, 1]]
)
    actual = solve_c48954c1(input_grid)
    assert actual == expected



def test_c48954c1_test_case_0():
    input_grid = ColoredGrid(values=
[[8, 8, 6], [6, 3, 6], [6, 8, 8]]
    )
    expected = ColoredGrid(values=
[[8, 8, 6, 6, 8, 8, 8, 8, 6],
 [6, 3, 6, 6, 3, 6, 6, 3, 6],
 [6, 8, 8, 8, 8, 6, 6, 8, 8],
 [6, 8, 8, 8, 8, 6, 6, 8, 8],
 [6, 3, 6, 6, 3, 6, 6, 3, 6],
 [8, 8, 6, 6, 8, 8, 8, 8, 6],
 [8, 8, 6, 6, 8, 8, 8, 8, 6],
 [6, 3, 6, 6, 3, 6, 6, 3, 6],
 [6, 8, 8, 8, 8, 6, 6, 8, 8]]
    )
    actual = solve_c48954c1(input_grid)
    assert actual == expected

