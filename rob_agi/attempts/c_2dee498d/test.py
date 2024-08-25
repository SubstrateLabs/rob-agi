import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_2dee498d.main import solve_2dee498d


def test_2dee498d_example_0():
    input_grid = ColoredGrid(values=
[[4, 5, 1, 1, 5, 4, 4, 5, 1],
 [5, 5, 5, 5, 5, 5, 5, 5, 5],
 [1, 5, 4, 4, 5, 1, 1, 5, 4]]
    )
    expected = ColoredGrid(values=
[[4, 5, 1], [5, 5, 5], [1, 5, 4]]
)
    actual = solve_2dee498d(input_grid)
    assert actual == expected


def test_2dee498d_example_1():
    input_grid = ColoredGrid(values=
[[2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1],
 [4, 2, 1, 4, 4, 2, 1, 4, 4, 2, 1, 4],
 [4, 1, 2, 4, 4, 1, 2, 4, 4, 1, 2, 4],
 [1, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0, 2]]
    )
    expected = ColoredGrid(values=
[[2, 0, 0, 1], [4, 2, 1, 4], [4, 1, 2, 4], [1, 0, 0, 2]]
)
    actual = solve_2dee498d(input_grid)
    assert actual == expected


def test_2dee498d_example_2():
    input_grid = ColoredGrid(values=
[[2, 1, 2, 1, 2, 1], [2, 3, 2, 3, 2, 3]]
    )
    expected = ColoredGrid(values=
[[2, 1], [2, 3]]
)
    actual = solve_2dee498d(input_grid)
    assert actual == expected



def test_2dee498d_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 2, 0, 4, 4, 0, 2, 0, 4, 4, 0, 2, 0, 4, 4],
 [2, 2, 0, 4, 4, 2, 2, 0, 4, 4, 2, 2, 0, 4, 4],
 [0, 2, 2, 2, 0, 0, 2, 2, 2, 0, 0, 2, 2, 2, 0],
 [1, 1, 0, 2, 2, 1, 1, 0, 2, 2, 1, 1, 0, 2, 2],
 [1, 1, 0, 2, 0, 1, 1, 0, 2, 0, 1, 1, 0, 2, 0]]
    )
    expected = ColoredGrid(values=
[[0, 2, 0, 4, 4],
 [2, 2, 0, 4, 4],
 [0, 2, 2, 2, 0],
 [1, 1, 0, 2, 2],
 [1, 1, 0, 2, 0]]
    )
    actual = solve_2dee498d(input_grid)
    assert actual == expected

