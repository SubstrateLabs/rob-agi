import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_b60334d2.main import solve_b60334d2

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_b60334d2_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 5, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 5, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 5, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 5, 1, 5, 0, 0, 0, 0],
 [0, 0, 1, 0, 1, 0, 0, 0, 0],
 [0, 0, 5, 1, 5, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 5, 1, 5, 0],
 [0, 0, 0, 0, 0, 1, 0, 1, 0],
 [0, 5, 1, 5, 0, 5, 1, 5, 0],
 [0, 1, 0, 1, 0, 0, 0, 0, 0],
 [0, 5, 1, 5, 0, 0, 0, 0, 0]]
)
    actual = solve_b60334d2(input_grid)
    assert actual == expected


def test_b60334d2_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 5, 0, 0, 0, 0, 5, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 5, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 5, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 5, 1, 5, 0, 0, 5, 1, 5],
 [0, 1, 0, 1, 0, 0, 1, 0, 1],
 [0, 5, 1, 5, 0, 0, 5, 1, 5],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 5, 1, 5, 0, 0, 0, 0, 0],
 [0, 1, 0, 1, 0, 0, 0, 0, 0],
 [0, 5, 1, 5, 0, 5, 1, 5, 0],
 [0, 0, 0, 0, 0, 1, 0, 1, 0],
 [0, 0, 0, 0, 0, 5, 1, 5, 0]]
)
    actual = solve_b60334d2(input_grid)
    assert actual == expected



