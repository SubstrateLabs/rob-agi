import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_4258a5f9.main import solve_4258a5f9

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_4258a5f9_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 5, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 5, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 5, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 1, 1, 1, 0],
 [0, 0, 0, 0, 0, 1, 5, 1, 0],
 [0, 0, 0, 0, 0, 1, 1, 1, 0],
 [0, 0, 1, 1, 1, 0, 0, 0, 0],
 [0, 0, 1, 5, 1, 0, 0, 0, 0],
 [0, 0, 1, 1, 1, 0, 0, 0, 0],
 [1, 1, 1, 0, 0, 0, 0, 0, 0],
 [1, 5, 1, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_4258a5f9(input_grid)
    assert actual == expected


def test_4258a5f9_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 5, 0],
 [0, 0, 0, 5, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 5, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 5, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 1, 1, 1],
 [0, 0, 1, 1, 1, 0, 1, 5, 1],
 [0, 0, 1, 5, 1, 0, 1, 1, 1],
 [0, 0, 1, 1, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 1, 1],
 [0, 0, 0, 0, 0, 0, 1, 5, 1],
 [0, 0, 1, 1, 1, 0, 1, 1, 1],
 [0, 0, 1, 5, 1, 0, 0, 0, 0],
 [0, 0, 1, 1, 1, 0, 0, 0, 0]]
)
    actual = solve_4258a5f9(input_grid)
    assert actual == expected



