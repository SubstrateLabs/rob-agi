import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_0ca9ddb6.main import solve_0ca9ddb6

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_0ca9ddb6_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 2, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 4, 0, 4, 0, 0, 0, 0, 0],
 [0, 0, 2, 0, 0, 0, 0, 0, 0],
 [0, 4, 0, 4, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 7, 0, 0],
 [0, 0, 0, 0, 0, 7, 1, 7, 0],
 [0, 0, 0, 0, 0, 0, 7, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_0ca9ddb6(input_grid)
    assert actual == expected


def test_0ca9ddb6_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 8, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 2, 0, 0],
 [0, 0, 1, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 0, 0],
 [0, 2, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 8, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 4, 0, 4, 0],
 [0, 0, 7, 0, 0, 0, 2, 0, 0],
 [0, 7, 1, 7, 0, 4, 0, 4, 0],
 [0, 0, 7, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 7, 0, 0],
 [4, 0, 4, 0, 0, 7, 1, 7, 0],
 [0, 2, 0, 0, 0, 0, 7, 0, 0],
 [4, 0, 4, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_0ca9ddb6(input_grid)
    assert actual == expected


def test_0ca9ddb6_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 2, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 6, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 4, 0, 4, 0, 0, 0, 0, 0],
 [0, 0, 2, 0, 0, 0, 0, 0, 0],
 [0, 4, 0, 4, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 6, 0, 0],
 [0, 0, 0, 7, 0, 0, 0, 0, 0],
 [0, 0, 7, 1, 7, 0, 0, 0, 0],
 [0, 0, 0, 7, 0, 0, 0, 0, 0]]
)
    actual = solve_0ca9ddb6(input_grid)
    assert actual == expected



