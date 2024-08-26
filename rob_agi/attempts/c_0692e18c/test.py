import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_0692e18c.main import solve_0692e18c

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_0692e18c_example_0():
    input_grid = ColoredGrid(values=
[[0, 7, 0], [7, 7, 7], [0, 7, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 7, 0, 7, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 7, 0, 7, 0, 0, 0],
 [7, 0, 7, 7, 0, 7, 7, 0, 7],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [7, 0, 7, 7, 0, 7, 7, 0, 7],
 [0, 0, 0, 7, 0, 7, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 7, 0, 7, 0, 0, 0]]
)
    actual = solve_0692e18c(input_grid)
    assert actual == expected


def test_0692e18c_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 6], [0, 6, 0], [6, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 6, 6, 0],
 [0, 0, 0, 0, 0, 0, 6, 0, 6],
 [0, 0, 0, 0, 0, 0, 0, 6, 6],
 [0, 0, 0, 6, 6, 0, 0, 0, 0],
 [0, 0, 0, 6, 0, 6, 0, 0, 0],
 [0, 0, 0, 0, 6, 6, 0, 0, 0],
 [6, 6, 0, 0, 0, 0, 0, 0, 0],
 [6, 0, 6, 0, 0, 0, 0, 0, 0],
 [0, 6, 6, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_0692e18c(input_grid)
    assert actual == expected


def test_0692e18c_example_2():
    input_grid = ColoredGrid(values=
[[4, 4, 0], [0, 0, 4], [0, 0, 4]]
    )
    expected = ColoredGrid(values=
[[0, 0, 4, 0, 0, 4, 0, 0, 0],
 [4, 4, 0, 4, 4, 0, 0, 0, 0],
 [4, 4, 0, 4, 4, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 4],
 [0, 0, 0, 0, 0, 0, 4, 4, 0],
 [0, 0, 0, 0, 0, 0, 4, 4, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 4],
 [0, 0, 0, 0, 0, 0, 4, 4, 0],
 [0, 0, 0, 0, 0, 0, 4, 4, 0]]
)
    actual = solve_0692e18c(input_grid)
    assert actual == expected



