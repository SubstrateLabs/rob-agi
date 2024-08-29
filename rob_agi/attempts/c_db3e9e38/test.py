import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_db3e9e38.main import solve_db3e9e38

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_db3e9e38_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 7, 8, 7, 8, 7, 8],
 [0, 7, 8, 7, 8, 7, 0],
 [0, 0, 8, 7, 8, 0, 0],
 [0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_db3e9e38(input_grid)
    assert actual == expected


def test_db3e9e38_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 7, 0, 0, 0, 0, 0],
 [0, 0, 7, 0, 0, 0, 0, 0],
 [0, 0, 7, 0, 0, 0, 0, 0],
 [0, 0, 7, 0, 0, 0, 0, 0],
 [0, 0, 7, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[7, 8, 7, 8, 7, 8, 7, 0],
 [7, 8, 7, 8, 7, 8, 0, 0],
 [7, 8, 7, 8, 7, 0, 0, 0],
 [0, 8, 7, 8, 0, 0, 0, 0],
 [0, 0, 7, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_db3e9e38(input_grid)
    assert actual == expected



