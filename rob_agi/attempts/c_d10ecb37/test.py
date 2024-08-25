import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_d10ecb37.main import solve_d10ecb37


def test_d10ecb37_example_0():
    input_grid = ColoredGrid(values=
[[4, 3, 6, 4, 0, 6],
 [6, 0, 0, 3, 3, 4],
 [6, 4, 4, 3, 3, 0],
 [0, 3, 6, 0, 4, 6],
 [0, 6, 3, 0, 4, 3],
 [3, 4, 4, 6, 6, 0]]
    )
    expected = ColoredGrid(values=
[[4, 3], [6, 0]]
)
    actual = solve_d10ecb37(input_grid)
    assert actual == expected


def test_d10ecb37_example_1():
    input_grid = ColoredGrid(values=
[[2, 4, 2, 2, 5, 2, 4, 5],
 [2, 5, 5, 4, 4, 2, 2, 2],
 [4, 5, 5, 2, 2, 2, 2, 4],
 [2, 2, 4, 2, 5, 4, 2, 5],
 [2, 4, 2, 2, 5, 2, 4, 5],
 [2, 5, 5, 4, 4, 2, 2, 2],
 [4, 5, 5, 2, 2, 2, 2, 4],
 [2, 2, 4, 2, 5, 4, 2, 5]]
    )
    expected = ColoredGrid(values=
[[2, 4], [2, 5]]
)
    actual = solve_d10ecb37(input_grid)
    assert actual == expected


def test_d10ecb37_example_2():
    input_grid = ColoredGrid(values=
[[3, 2, 1, 3, 4, 1],
 [1, 4, 4, 2, 2, 3],
 [1, 3, 3, 2, 2, 4],
 [4, 2, 1, 4, 3, 1],
 [4, 1, 2, 4, 3, 2],
 [2, 3, 3, 1, 1, 4],
 [2, 4, 4, 1, 1, 3],
 [3, 1, 2, 3, 4, 2],
 [3, 2, 1, 3, 4, 1],
 [1, 4, 4, 2, 2, 3],
 [1, 3, 3, 2, 2, 4],
 [4, 2, 1, 4, 3, 1]]
    )
    expected = ColoredGrid(values=
[[3, 2], [1, 4]]
)
    actual = solve_d10ecb37(input_grid)
    assert actual == expected



def test_d10ecb37_test_case_0():
    input_grid = ColoredGrid(values=
[[9, 6, 2, 9, 9, 2, 6, 9],
 [2, 9, 9, 6, 6, 9, 9, 2],
 [6, 9, 9, 2, 2, 9, 9, 6],
 [9, 2, 6, 9, 9, 6, 2, 9]]
    )
    expected = ColoredGrid(values=
[[9, 6], [2, 9]]
    )
    actual = solve_d10ecb37(input_grid)
    assert actual == expected

