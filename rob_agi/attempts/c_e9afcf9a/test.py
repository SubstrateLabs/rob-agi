import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_e9afcf9a.main import solve_e9afcf9a

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_e9afcf9a_example_0():
    input_grid = ColoredGrid(values=
[[3, 3, 3, 3, 3, 3], [9, 9, 9, 9, 9, 9]]
    )
    expected = ColoredGrid(values=
[[3, 9, 3, 9, 3, 9], [9, 3, 9, 3, 9, 3]]
)
    actual = solve_e9afcf9a(input_grid)
    assert actual == expected


def test_e9afcf9a_example_1():
    input_grid = ColoredGrid(values=
[[4, 4, 4, 4, 4, 4], [8, 8, 8, 8, 8, 8]]
    )
    expected = ColoredGrid(values=
[[4, 8, 4, 8, 4, 8], [8, 4, 8, 4, 8, 4]]
)
    actual = solve_e9afcf9a(input_grid)
    assert actual == expected



