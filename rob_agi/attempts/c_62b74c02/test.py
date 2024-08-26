import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_62b74c02.main import solve_62b74c02

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_62b74c02_example_0():
    input_grid = ColoredGrid(values=
[[8, 2, 8, 0, 0, 0, 0, 0, 0, 0],
 [1, 8, 1, 0, 0, 0, 0, 0, 0, 0],
 [1, 8, 1, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 2, 8, 8, 8, 8, 8, 8, 2, 8],
 [1, 8, 1, 1, 1, 1, 1, 1, 8, 1],
 [1, 8, 1, 1, 1, 1, 1, 1, 8, 1]]
)
    actual = solve_62b74c02(input_grid)
    assert actual == expected


def test_62b74c02_example_1():
    input_grid = ColoredGrid(values=
[[3, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [3, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3],
 [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
 [3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3],
 [1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1]]
)
    actual = solve_62b74c02(input_grid)
    assert actual == expected


def test_62b74c02_example_2():
    input_grid = ColoredGrid(values=
[[2, 3, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [2, 8, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [2, 8, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 3, 8, 2, 2, 2, 2, 2, 2, 2, 2, 3, 8, 2],
 [2, 8, 8, 2, 2, 2, 2, 2, 2, 2, 2, 8, 8, 2],
 [2, 8, 3, 2, 2, 2, 2, 2, 2, 2, 2, 8, 3, 2]]
)
    actual = solve_62b74c02(input_grid)
    assert actual == expected



