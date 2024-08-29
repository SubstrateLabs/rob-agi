import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_d0f5fe59.main import solve_d0f5fe59

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_d0f5fe59_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 8, 8, 0, 0, 0, 0, 0, 0],
 [0, 0, 8, 8, 0, 0, 0, 0, 0, 0],
 [0, 8, 8, 8, 0, 0, 0, 0, 0, 0],
 [0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 8, 8, 0, 0, 0],
 [0, 0, 0, 0, 8, 8, 8, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 8, 8, 0, 0, 0, 0, 0],
 [0, 8, 8, 8, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 8, 0, 0, 0, 8, 8, 0],
 [0, 0, 0, 0, 0, 0, 0, 8, 8, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 0, 0, 0], [0, 8, 0, 0], [0, 0, 8, 0], [0, 0, 0, 8]]
)
    actual = solve_d0f5fe59(input_grid)
    assert actual == expected


def test_d0f5fe59_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 8, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0],
 [0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 0, 0], [0, 8, 0], [0, 0, 8]]
)
    actual = solve_d0f5fe59(input_grid)
    assert actual == expected


def test_d0f5fe59_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 8, 8, 8, 0, 0, 0, 8, 0, 0, 0],
 [0, 0, 0, 8, 8, 0, 0, 0, 8, 8, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 0], [0, 8]]
)
    actual = solve_d0f5fe59(input_grid)
    assert actual == expected



