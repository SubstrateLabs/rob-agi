import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_b7cb93ac.main import solve_b7cb93ac

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_b7cb93ac_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 2, 2, 1], [1, 1, 1, 1], [1, 8, 8, 1]]
)
    actual = solve_b7cb93ac(input_grid)
    assert actual == expected


def test_b7cb93ac_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0],
 [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 3, 3, 3], [2, 2, 2, 3], [2, 8, 3, 3]]
)
    actual = solve_b7cb93ac(input_grid)
    assert actual == expected


def test_b7cb93ac_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
 [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
 [0, 8, 0, 0, 0, 0, 0, 0, 2, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 1, 2, 2], [1, 1, 1, 1], [8, 1, 2, 2]]
)
    actual = solve_b7cb93ac(input_grid)
    assert actual == expected



