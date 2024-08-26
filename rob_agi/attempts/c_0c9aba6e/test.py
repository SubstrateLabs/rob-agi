import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_0c9aba6e.main import solve_0c9aba6e

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_0c9aba6e_example_0():
    input_grid = ColoredGrid(values=
[[0, 2, 2, 0],
 [2, 0, 0, 0],
 [0, 2, 0, 2],
 [2, 2, 2, 2],
 [0, 0, 2, 0],
 [0, 0, 2, 2],
 [7, 7, 7, 7],
 [0, 6, 6, 0],
 [0, 0, 0, 0],
 [6, 6, 6, 6],
 [6, 6, 0, 6],
 [0, 6, 6, 6],
 [0, 0, 6, 0]]
    )
    expected = ColoredGrid(values=
[[8, 0, 0, 8],
 [0, 8, 8, 8],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [8, 0, 0, 0],
 [8, 8, 0, 0]]
)
    actual = solve_0c9aba6e(input_grid)
    assert actual == expected


def test_0c9aba6e_example_1():
    input_grid = ColoredGrid(values=
[[2, 2, 0, 2],
 [2, 0, 2, 2],
 [2, 2, 0, 0],
 [0, 2, 0, 2],
 [0, 2, 2, 0],
 [2, 0, 0, 2],
 [7, 7, 7, 7],
 [6, 0, 6, 6],
 [0, 6, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 6],
 [6, 6, 0, 0],
 [6, 0, 6, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 8, 8],
 [8, 0, 8, 0],
 [0, 0, 0, 8],
 [0, 8, 0, 0]]
)
    actual = solve_0c9aba6e(input_grid)
    assert actual == expected


def test_0c9aba6e_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 2],
 [2, 0, 0, 0],
 [0, 2, 2, 2],
 [0, 0, 0, 2],
 [2, 0, 2, 0],
 [0, 2, 2, 0],
 [7, 7, 7, 7],
 [6, 0, 6, 6],
 [6, 0, 0, 6],
 [0, 6, 6, 6],
 [6, 0, 0, 0],
 [6, 0, 0, 6],
 [0, 0, 6, 0]]
    )
    expected = ColoredGrid(values=
[[0, 8, 0, 0],
 [0, 8, 8, 0],
 [8, 0, 0, 0],
 [0, 8, 8, 0],
 [0, 8, 0, 0],
 [8, 0, 0, 8]]
)
    actual = solve_0c9aba6e(input_grid)
    assert actual == expected


def test_0c9aba6e_example_3():
    input_grid = ColoredGrid(values=
[[2, 2, 0, 0],
 [0, 2, 2, 0],
 [2, 2, 0, 0],
 [2, 0, 0, 0],
 [0, 0, 0, 2],
 [2, 2, 0, 0],
 [7, 7, 7, 7],
 [6, 6, 6, 6],
 [6, 0, 6, 6],
 [6, 6, 0, 0],
 [0, 0, 0, 0],
 [6, 6, 0, 0],
 [0, 0, 6, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 8, 8],
 [0, 8, 8, 8],
 [0, 0, 8, 0],
 [0, 0, 0, 8]]
)
    actual = solve_0c9aba6e(input_grid)
    assert actual == expected



