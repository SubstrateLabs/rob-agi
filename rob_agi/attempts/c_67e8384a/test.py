import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_67e8384a.main import solve_67e8384a


def test_67e8384a_example_0():
    input_grid = ColoredGrid(values=
[[5, 3, 4], [3, 4, 5], [3, 4, 4]]
    )
    expected = ColoredGrid(values=
[[5, 3, 4, 4, 3, 5],
 [3, 4, 5, 5, 4, 3],
 [3, 4, 4, 4, 4, 3],
 [3, 4, 4, 4, 4, 3],
 [3, 4, 5, 5, 4, 3],
 [5, 3, 4, 4, 3, 5]]
)
    actual = solve_67e8384a(input_grid)
    assert actual == expected


def test_67e8384a_example_1():
    input_grid = ColoredGrid(values=
[[7, 1, 5], [7, 7, 1], [5, 3, 1]]
    )
    expected = ColoredGrid(values=
[[7, 1, 5, 5, 1, 7],
 [7, 7, 1, 1, 7, 7],
 [5, 3, 1, 1, 3, 5],
 [5, 3, 1, 1, 3, 5],
 [7, 7, 1, 1, 7, 7],
 [7, 1, 5, 5, 1, 7]]
)
    actual = solve_67e8384a(input_grid)
    assert actual == expected


def test_67e8384a_example_2():
    input_grid = ColoredGrid(values=
[[2, 5, 2], [2, 6, 4], [2, 2, 2]]
    )
    expected = ColoredGrid(values=
[[2, 5, 2, 2, 5, 2],
 [2, 6, 4, 4, 6, 2],
 [2, 2, 2, 2, 2, 2],
 [2, 2, 2, 2, 2, 2],
 [2, 6, 4, 4, 6, 2],
 [2, 5, 2, 2, 5, 2]]
)
    actual = solve_67e8384a(input_grid)
    assert actual == expected


def test_67e8384a_example_3():
    input_grid = ColoredGrid(values=
[[1, 2, 1], [2, 8, 1], [8, 1, 6]]
    )
    expected = ColoredGrid(values=
[[1, 2, 1, 1, 2, 1],
 [2, 8, 1, 1, 8, 2],
 [8, 1, 6, 6, 1, 8],
 [8, 1, 6, 6, 1, 8],
 [2, 8, 1, 1, 8, 2],
 [1, 2, 1, 1, 2, 1]]
)
    actual = solve_67e8384a(input_grid)
    assert actual == expected



def test_67e8384a_test_case_0():
    input_grid = ColoredGrid(values=
[[1, 6, 6], [5, 2, 2], [2, 2, 2]]
    )
    expected = ColoredGrid(values=
[[1, 6, 6, 6, 6, 1],
 [5, 2, 2, 2, 2, 5],
 [2, 2, 2, 2, 2, 2],
 [2, 2, 2, 2, 2, 2],
 [5, 2, 2, 2, 2, 5],
 [1, 6, 6, 6, 6, 1]]
    )
    actual = solve_67e8384a(input_grid)
    assert actual == expected

