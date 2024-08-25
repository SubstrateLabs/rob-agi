import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_99b1bc43.main import solve_99b1bc43


def test_99b1bc43_example_0():
    input_grid = ColoredGrid(values=
[[0, 1, 0, 1],
 [0, 0, 0, 1],
 [1, 0, 1, 0],
 [0, 0, 0, 1],
 [4, 4, 4, 4],
 [0, 2, 0, 2],
 [0, 0, 0, 2],
 [2, 0, 0, 2],
 [2, 2, 2, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 3, 3], [3, 3, 3, 3]]
)
    actual = solve_99b1bc43(input_grid)
    assert actual == expected


def test_99b1bc43_example_1():
    input_grid = ColoredGrid(values=
[[1, 1, 0, 0],
 [1, 0, 1, 0],
 [1, 1, 0, 1],
 [0, 1, 1, 0],
 [4, 4, 4, 4],
 [0, 2, 2, 2],
 [2, 0, 2, 0],
 [2, 2, 2, 2],
 [2, 2, 2, 2]]
    )
    expected = ColoredGrid(values=
[[3, 0, 3, 3], [0, 0, 0, 0], [0, 0, 3, 0], [3, 0, 0, 3]]
)
    actual = solve_99b1bc43(input_grid)
    assert actual == expected


def test_99b1bc43_example_2():
    input_grid = ColoredGrid(values=
[[0, 1, 0, 0],
 [1, 0, 1, 1],
 [1, 1, 1, 0],
 [1, 1, 1, 0],
 [4, 4, 4, 4],
 [0, 0, 0, 0],
 [0, 2, 0, 2],
 [2, 2, 0, 2],
 [0, 2, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 3, 0, 0], [3, 3, 3, 0], [0, 0, 3, 3], [3, 0, 3, 0]]
)
    actual = solve_99b1bc43(input_grid)
    assert actual == expected


def test_99b1bc43_example_3():
    input_grid = ColoredGrid(values=
[[1, 0, 1, 1],
 [0, 0, 0, 1],
 [1, 1, 0, 0],
 [0, 0, 1, 1],
 [4, 4, 4, 4],
 [0, 2, 2, 2],
 [0, 2, 2, 2],
 [2, 0, 2, 2],
 [2, 2, 2, 2]]
    )
    expected = ColoredGrid(values=
[[3, 3, 0, 0], [0, 3, 3, 0], [0, 3, 3, 3], [3, 3, 0, 0]]
)
    actual = solve_99b1bc43(input_grid)
    assert actual == expected



def test_99b1bc43_test_case_0():
    input_grid = ColoredGrid(values=
[[1, 0, 1, 1],
 [0, 1, 1, 1],
 [0, 0, 1, 0],
 [1, 0, 1, 1],
 [4, 4, 4, 4],
 [2, 2, 0, 2],
 [0, 0, 2, 0],
 [2, 0, 0, 2],
 [0, 2, 0, 2]]
    )
    expected = ColoredGrid(values=
[[0, 3, 3, 0], [0, 3, 0, 3], [3, 0, 3, 3], [3, 3, 3, 0]]
    )
    actual = solve_99b1bc43(input_grid)
    assert actual == expected

