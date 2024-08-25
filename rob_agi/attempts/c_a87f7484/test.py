import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_a87f7484.main import solve_a87f7484


def test_a87f7484_example_0():
    input_grid = ColoredGrid(values=
[[6, 0, 6],
 [0, 6, 6],
 [6, 0, 6],
 [4, 0, 4],
 [0, 4, 4],
 [4, 0, 4],
 [8, 8, 8],
 [8, 0, 8],
 [8, 8, 8]]
    )
    expected = ColoredGrid(values=
[[8, 8, 8], [8, 0, 8], [8, 8, 8]]
)
    actual = solve_a87f7484(input_grid)
    assert actual == expected


def test_a87f7484_example_1():
    input_grid = ColoredGrid(values=
[[2, 0, 0, 3, 0, 0, 7, 0, 7, 1, 0, 0],
 [2, 0, 0, 3, 0, 0, 0, 7, 0, 1, 0, 0],
 [0, 2, 2, 0, 3, 3, 7, 0, 7, 0, 1, 1]]
    )
    expected = ColoredGrid(values=
[[7, 0, 7], [0, 7, 0], [7, 0, 7]]
)
    actual = solve_a87f7484(input_grid)
    assert actual == expected


def test_a87f7484_example_2():
    input_grid = ColoredGrid(values=
[[3, 0, 0, 4, 0, 4, 2, 0, 0, 8, 0, 0, 1, 0, 0],
 [0, 3, 3, 4, 4, 4, 0, 2, 2, 0, 8, 8, 0, 1, 1],
 [0, 3, 0, 4, 0, 4, 0, 2, 0, 0, 8, 0, 0, 1, 0]]
    )
    expected = ColoredGrid(values=
[[4, 0, 4], [4, 4, 4], [4, 0, 4]]
)
    actual = solve_a87f7484(input_grid)
    assert actual == expected


def test_a87f7484_example_3():
    input_grid = ColoredGrid(values=
[[0, 7, 7],
 [7, 7, 0],
 [7, 0, 7],
 [3, 0, 0],
 [0, 3, 3],
 [3, 0, 0],
 [2, 0, 0],
 [0, 2, 2],
 [2, 0, 0],
 [8, 0, 0],
 [0, 8, 8],
 [8, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 7, 7], [7, 7, 0], [7, 0, 7]]
)
    actual = solve_a87f7484(input_grid)
    assert actual == expected



def test_a87f7484_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 5, 0],
 [5, 0, 5],
 [0, 5, 0],
 [0, 3, 0],
 [3, 0, 3],
 [0, 3, 0],
 [6, 0, 6],
 [6, 6, 0],
 [6, 0, 6],
 [0, 4, 0],
 [4, 0, 4],
 [0, 4, 0],
 [0, 8, 0],
 [8, 0, 8],
 [0, 8, 0]]
    )
    expected = ColoredGrid(values=
[[6, 0, 6], [6, 6, 0], [6, 0, 6]]
    )
    actual = solve_a87f7484(input_grid)
    assert actual == expected

