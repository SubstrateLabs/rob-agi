import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_2a5f8217.main import solve_2a5f8217

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_2a5f8217_example_0():
    input_grid = ColoredGrid(values=
[[0, 1, 0, 0, 0, 0],
 [1, 1, 1, 0, 0, 0],
 [0, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 8, 0],
 [0, 0, 0, 8, 8, 8],
 [0, 0, 0, 0, 8, 0]]
    )
    expected = ColoredGrid(values=
[[0, 8, 0, 0, 0, 0],
 [8, 8, 8, 0, 0, 0],
 [0, 8, 0, 0, 0, 0],
 [0, 0, 0, 0, 8, 0],
 [0, 0, 0, 8, 8, 8],
 [0, 0, 0, 0, 8, 0]]
)
    actual = solve_2a5f8217(input_grid)
    assert actual == expected


def test_2a5f8217_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 1, 1, 0, 0, 1, 0, 0],
 [0, 1, 0, 1, 0, 0, 1, 1, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 9, 9],
 [0, 0, 1, 1, 0, 0, 0, 0, 9],
 [0, 0, 0, 1, 0, 7, 0, 0, 0],
 [6, 6, 6, 0, 0, 7, 7, 0, 0],
 [6, 0, 6, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 6, 6, 6, 0, 0, 7, 0, 0],
 [0, 6, 0, 6, 0, 0, 7, 7, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 9, 9],
 [0, 0, 9, 9, 0, 0, 0, 0, 9],
 [0, 0, 0, 9, 0, 7, 0, 0, 0],
 [6, 6, 6, 0, 0, 7, 7, 0, 0],
 [6, 0, 6, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_2a5f8217(input_grid)
    assert actual == expected


def test_2a5f8217_example_2():
    input_grid = ColoredGrid(values=
[[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 9],
 [1, 1, 1, 0, 0, 1, 0, 0, 0, 9, 9],
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9],
 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
 [0, 3, 0, 0, 0, 6, 0, 1, 1, 0, 0],
 [3, 3, 3, 0, 0, 6, 0, 0, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 7, 7, 7, 0],
 [0, 0, 1, 1, 1, 0, 0, 0, 0, 7, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 7, 0, 0, 0, 6, 0, 0, 0, 0, 9],
 [7, 7, 7, 0, 0, 6, 0, 0, 0, 9, 9],
 [0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 9],
 [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
 [0, 3, 0, 0, 0, 6, 0, 9, 9, 0, 0],
 [3, 3, 3, 0, 0, 6, 0, 0, 9, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0],
 [0, 0, 0, 3, 0, 0, 0, 7, 7, 7, 0],
 [0, 0, 3, 3, 3, 0, 0, 0, 0, 7, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_2a5f8217(input_grid)
    assert actual == expected



