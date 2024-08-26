import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_9110e3c5.main import solve_9110e3c5

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_9110e3c5_example_0():
    input_grid = ColoredGrid(values=
[[0, 4, 1, 0, 0, 1, 6],
 [0, 0, 1, 0, 0, 0, 0],
 [1, 1, 0, 0, 1, 1, 0],
 [0, 1, 0, 0, 0, 1, 1],
 [0, 0, 1, 0, 0, 2, 0],
 [1, 0, 1, 0, 1, 0, 7],
 [1, 1, 1, 0, 4, 1, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 8], [8, 8, 0], [0, 8, 0]]
)
    actual = solve_9110e3c5(input_grid)
    assert actual == expected


def test_9110e3c5_example_1():
    input_grid = ColoredGrid(values=
[[2, 0, 0, 2, 2, 0, 5],
 [0, 2, 2, 0, 0, 0, 2],
 [0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 2, 0, 9],
 [0, 9, 0, 0, 0, 0, 2],
 [0, 0, 2, 1, 0, 0, 8],
 [2, 0, 0, 2, 2, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [8, 8, 8], [0, 0, 0]]
)
    actual = solve_9110e3c5(input_grid)
    assert actual == expected


def test_9110e3c5_example_2():
    input_grid = ColoredGrid(values=
[[0, 4, 0, 0, 4, 1, 3],
 [3, 3, 4, 3, 0, 3, 7],
 [3, 0, 0, 0, 1, 0, 3],
 [0, 0, 3, 0, 3, 0, 0],
 [3, 0, 0, 3, 3, 0, 3],
 [3, 0, 3, 0, 3, 0, 3],
 [3, 3, 3, 0, 4, 2, 3]]
    )
    expected = ColoredGrid(values=
[[0, 8, 8], [0, 8, 0], [0, 8, 0]]
)
    actual = solve_9110e3c5(input_grid)
    assert actual == expected


def test_9110e3c5_example_3():
    input_grid = ColoredGrid(values=
[[1, 0, 1, 0, 7, 0, 0],
 [1, 1, 9, 1, 0, 1, 0],
 [0, 0, 1, 1, 0, 2, 0],
 [0, 0, 0, 0, 3, 0, 1],
 [0, 4, 0, 1, 0, 0, 1],
 [0, 0, 1, 0, 2, 0, 8],
 [0, 0, 1, 0, 7, 3, 1]]
    )
    expected = ColoredGrid(values=
[[0, 0, 8], [8, 8, 0], [0, 8, 0]]
)
    actual = solve_9110e3c5(input_grid)
    assert actual == expected


def test_9110e3c5_example_4():
    input_grid = ColoredGrid(values=
[[0, 3, 0, 3, 5, 3, 0],
 [0, 0, 3, 3, 0, 0, 0],
 [8, 0, 0, 0, 0, 0, 3],
 [3, 4, 3, 9, 3, 0, 3],
 [0, 0, 9, 3, 1, 3, 3],
 [0, 3, 3, 3, 0, 3, 0],
 [0, 0, 0, 0, 0, 0, 3]]
    )
    expected = ColoredGrid(values=
[[0, 8, 8], [0, 8, 0], [0, 8, 0]]
)
    actual = solve_9110e3c5(input_grid)
    assert actual == expected


def test_9110e3c5_example_5():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 2, 2, 0, 2],
 [0, 2, 2, 9, 2, 2, 0],
 [0, 5, 0, 2, 4, 6, 0],
 [2, 0, 0, 0, 0, 9, 2],
 [0, 0, 0, 2, 2, 0, 0],
 [8, 0, 2, 9, 0, 6, 3],
 [0, 2, 0, 2, 0, 2, 4]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [8, 8, 8], [0, 0, 0]]
)
    actual = solve_9110e3c5(input_grid)
    assert actual == expected


def test_9110e3c5_example_6():
    input_grid = ColoredGrid(values=
[[0, 0, 2, 0, 1, 5, 3],
 [0, 0, 2, 9, 0, 2, 0],
 [2, 2, 2, 4, 2, 0, 0],
 [0, 2, 0, 2, 7, 2, 0],
 [2, 2, 0, 0, 2, 2, 6],
 [0, 2, 2, 0, 2, 0, 0],
 [5, 0, 4, 2, 0, 2, 2]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [8, 8, 8], [0, 0, 0]]
)
    actual = solve_9110e3c5(input_grid)
    assert actual == expected



