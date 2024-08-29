import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_5582e5ca.main import solve_5582e5ca

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_5582e5ca_example_0():
    input_grid = ColoredGrid(values=
[[4, 4, 8], [6, 4, 3], [6, 3, 0]]
    )
    expected = ColoredGrid(values=
[[4, 4, 4], [4, 4, 4], [4, 4, 4]]
)
    actual = solve_5582e5ca(input_grid)
    assert actual == expected


def test_5582e5ca_example_1():
    input_grid = ColoredGrid(values=
[[6, 8, 9], [1, 8, 1], [9, 4, 9]]
    )
    expected = ColoredGrid(values=
[[9, 9, 9], [9, 9, 9], [9, 9, 9]]
)
    actual = solve_5582e5ca(input_grid)
    assert actual == expected


def test_5582e5ca_example_2():
    input_grid = ColoredGrid(values=
[[4, 6, 9], [6, 4, 1], [8, 8, 6]]
    )
    expected = ColoredGrid(values=
[[6, 6, 6], [6, 6, 6], [6, 6, 6]]
)
    actual = solve_5582e5ca(input_grid)
    assert actual == expected



