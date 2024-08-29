import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_23581191.main import solve_23581191

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_23581191_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 8, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 7, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 8, 0, 0, 0, 7, 0, 0],
 [0, 0, 8, 0, 0, 0, 7, 0, 0],
 [8, 8, 8, 8, 8, 8, 2, 8, 8],
 [0, 0, 8, 0, 0, 0, 7, 0, 0],
 [0, 0, 8, 0, 0, 0, 7, 0, 0],
 [0, 0, 8, 0, 0, 0, 7, 0, 0],
 [7, 7, 2, 7, 7, 7, 7, 7, 7],
 [0, 0, 8, 0, 0, 0, 7, 0, 0],
 [0, 0, 8, 0, 0, 0, 7, 0, 0]]
)
    actual = solve_23581191(input_grid)
    assert actual == expected


def test_23581191_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 8, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 7, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 8, 0, 0, 7, 0, 0],
 [8, 8, 8, 8, 8, 8, 2, 8, 8],
 [0, 0, 0, 8, 0, 0, 7, 0, 0],
 [0, 0, 0, 8, 0, 0, 7, 0, 0],
 [0, 0, 0, 8, 0, 0, 7, 0, 0],
 [0, 0, 0, 8, 0, 0, 7, 0, 0],
 [0, 0, 0, 8, 0, 0, 7, 0, 0],
 [7, 7, 7, 2, 7, 7, 7, 7, 7],
 [0, 0, 0, 8, 0, 0, 7, 0, 0]]
)
    actual = solve_23581191(input_grid)
    assert actual == expected



