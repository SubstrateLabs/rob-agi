import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_1c0d0a4b.main import solve_1c0d0a4b

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_1c0d0a4b_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 8, 0, 8, 0, 8, 0, 0, 0, 8, 8, 0, 0],
 [0, 8, 0, 8, 0, 0, 8, 0, 0, 0, 8, 0, 0],
 [0, 8, 8, 8, 0, 8, 8, 8, 0, 0, 8, 8, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 0],
 [0, 0, 2, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_1c0d0a4b(input_grid)
    assert actual == expected


def test_1c0d0a4b_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 0, 0],
 [0, 0, 8, 0, 0, 8, 8, 8, 0, 8, 8, 0, 0],
 [0, 8, 0, 8, 0, 8, 0, 8, 0, 0, 8, 8, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 8, 8, 8, 0, 8, 8, 8, 0, 8, 8, 0, 0],
 [0, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0],
 [0, 0, 8, 0, 0, 8, 8, 8, 0, 8, 8, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 2, 0],
 [0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0],
 [0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
 [0, 0, 2, 0, 0, 2, 2, 2, 0, 2, 2, 0, 0],
 [0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_1c0d0a4b(input_grid)
    assert actual == expected


def test_1c0d0a4b_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 8, 8, 8, 0, 8, 0, 8, 0],
 [0, 0, 0, 0, 0, 0, 8, 8, 0],
 [0, 0, 8, 0, 0, 8, 0, 8, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 8, 8, 0, 0, 8, 0, 0, 0],
 [0, 0, 8, 8, 0, 8, 8, 8, 0],
 [0, 8, 0, 0, 0, 0, 8, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 2, 0, 0],
 [0, 2, 2, 2, 0, 2, 0, 0, 0],
 [0, 2, 0, 2, 0, 0, 2, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 2, 0, 0, 2, 2, 0],
 [0, 2, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 2, 2, 0, 2, 0, 2, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_1c0d0a4b(input_grid)
    assert actual == expected



