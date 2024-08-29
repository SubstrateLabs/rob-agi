import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_9172f3a0.main import solve_9172f3a0

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_9172f3a0_example_0():
    input_grid = ColoredGrid(values=
[[3, 3, 0], [7, 4, 0], [0, 0, 4]]
    )
    expected = ColoredGrid(values=
[[3, 3, 3, 3, 3, 3, 0, 0, 0],
 [3, 3, 3, 3, 3, 3, 0, 0, 0],
 [3, 3, 3, 3, 3, 3, 0, 0, 0],
 [7, 7, 7, 4, 4, 4, 0, 0, 0],
 [7, 7, 7, 4, 4, 4, 0, 0, 0],
 [7, 7, 7, 4, 4, 4, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 4, 4, 4],
 [0, 0, 0, 0, 0, 0, 4, 4, 4],
 [0, 0, 0, 0, 0, 0, 4, 4, 4]]
)
    actual = solve_9172f3a0(input_grid)
    assert actual == expected


def test_9172f3a0_example_1():
    input_grid = ColoredGrid(values=
[[3, 0, 2], [0, 2, 2], [0, 0, 3]]
    )
    expected = ColoredGrid(values=
[[3, 3, 3, 0, 0, 0, 2, 2, 2],
 [3, 3, 3, 0, 0, 0, 2, 2, 2],
 [3, 3, 3, 0, 0, 0, 2, 2, 2],
 [0, 0, 0, 2, 2, 2, 2, 2, 2],
 [0, 0, 0, 2, 2, 2, 2, 2, 2],
 [0, 0, 0, 2, 2, 2, 2, 2, 2],
 [0, 0, 0, 0, 0, 0, 3, 3, 3],
 [0, 0, 0, 0, 0, 0, 3, 3, 3],
 [0, 0, 0, 0, 0, 0, 3, 3, 3]]
)
    actual = solve_9172f3a0(input_grid)
    assert actual == expected



