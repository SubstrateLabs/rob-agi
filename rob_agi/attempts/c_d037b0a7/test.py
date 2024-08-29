import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_d037b0a7.main import solve_d037b0a7

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_d037b0a7_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 6], [0, 4, 0], [3, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 6], [0, 4, 6], [3, 4, 6]]
)
    actual = solve_d037b0a7(input_grid)
    assert actual == expected


def test_d037b0a7_example_1():
    input_grid = ColoredGrid(values=
[[0, 2, 0], [7, 0, 8], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 2, 0], [7, 2, 8], [7, 2, 8]]
)
    actual = solve_d037b0a7(input_grid)
    assert actual == expected


def test_d037b0a7_example_2():
    input_grid = ColoredGrid(values=
[[4, 0, 0], [0, 2, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[4, 0, 0], [4, 2, 0], [4, 2, 0]]
)
    actual = solve_d037b0a7(input_grid)
    assert actual == expected



