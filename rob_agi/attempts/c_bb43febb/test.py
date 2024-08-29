import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_bb43febb.main import solve_bb43febb

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_bb43febb_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [5, 5, 5, 5, 5, 0, 0, 0, 0, 0],
 [5, 5, 5, 5, 5, 0, 5, 5, 5, 0],
 [5, 5, 5, 5, 5, 0, 5, 5, 5, 0],
 [5, 5, 5, 5, 5, 0, 5, 5, 5, 0],
 [5, 5, 5, 5, 5, 0, 5, 5, 5, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [5, 5, 5, 5, 5, 0, 0, 0, 0, 0],
 [5, 2, 2, 2, 5, 0, 5, 5, 5, 0],
 [5, 2, 2, 2, 5, 0, 5, 2, 5, 0],
 [5, 2, 2, 2, 5, 0, 5, 2, 5, 0],
 [5, 5, 5, 5, 5, 0, 5, 5, 5, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_bb43febb(input_grid)
    assert actual == expected


def test_bb43febb_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 5, 5, 5, 5, 5, 5, 0, 0, 0],
 [0, 5, 5, 5, 5, 5, 5, 0, 0, 0],
 [0, 5, 5, 5, 5, 5, 5, 0, 0, 0],
 [0, 5, 5, 5, 5, 5, 5, 0, 0, 0],
 [0, 5, 5, 5, 5, 5, 5, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 5, 5, 5, 5, 5, 0],
 [0, 0, 0, 0, 5, 5, 5, 5, 5, 0],
 [0, 0, 0, 0, 5, 5, 5, 5, 5, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 5, 5, 5, 5, 5, 5, 0, 0, 0],
 [0, 5, 2, 2, 2, 2, 5, 0, 0, 0],
 [0, 5, 2, 2, 2, 2, 5, 0, 0, 0],
 [0, 5, 2, 2, 2, 2, 5, 0, 0, 0],
 [0, 5, 5, 5, 5, 5, 5, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 5, 5, 5, 5, 5, 0],
 [0, 0, 0, 0, 5, 2, 2, 2, 5, 0],
 [0, 0, 0, 0, 5, 5, 5, 5, 5, 0]]
)
    actual = solve_bb43febb(input_grid)
    assert actual == expected



