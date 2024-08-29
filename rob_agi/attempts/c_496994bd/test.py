import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_496994bd.main import solve_496994bd

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_496994bd_example_0():
    input_grid = ColoredGrid(values=
[[2, 2, 2],
 [2, 2, 2],
 [3, 3, 3],
 [0, 0, 0],
 [0, 0, 0],
 [0, 0, 0],
 [0, 0, 0],
 [0, 0, 0],
 [0, 0, 0],
 [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 2, 2],
 [2, 2, 2],
 [3, 3, 3],
 [0, 0, 0],
 [0, 0, 0],
 [0, 0, 0],
 [0, 0, 0],
 [3, 3, 3],
 [2, 2, 2],
 [2, 2, 2]]
)
    actual = solve_496994bd(input_grid)
    assert actual == expected


def test_496994bd_example_1():
    input_grid = ColoredGrid(values=
[[2, 2, 2, 2, 2],
 [8, 8, 8, 8, 8],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 2, 2, 2, 2],
 [8, 8, 8, 8, 8],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [8, 8, 8, 8, 8],
 [2, 2, 2, 2, 2]]
)
    actual = solve_496994bd(input_grid)
    assert actual == expected



