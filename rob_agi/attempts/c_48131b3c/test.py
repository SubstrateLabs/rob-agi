import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_48131b3c.main import solve_48131b3c

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_48131b3c_example_0():
    input_grid = ColoredGrid(values=
[[0, 8, 0], [8, 0, 8], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 0, 8, 8, 0, 8],
 [0, 8, 0, 0, 8, 0],
 [8, 8, 8, 8, 8, 8],
 [8, 0, 8, 8, 0, 8],
 [0, 8, 0, 0, 8, 0],
 [8, 8, 8, 8, 8, 8]]
)
    actual = solve_48131b3c(input_grid)
    assert actual == expected


def test_48131b3c_example_1():
    input_grid = ColoredGrid(values=
[[7, 0], [0, 7]]
    )
    expected = ColoredGrid(values=
[[0, 7, 0, 7], [7, 0, 7, 0], [0, 7, 0, 7], [7, 0, 7, 0]]
)
    actual = solve_48131b3c(input_grid)
    assert actual == expected


def test_48131b3c_example_2():
    input_grid = ColoredGrid(values=
[[4, 0, 0, 0], [0, 4, 4, 4], [0, 0, 4, 0], [0, 4, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 4, 4, 4, 0, 4, 4, 4],
 [4, 0, 0, 0, 4, 0, 0, 0],
 [4, 4, 0, 4, 4, 4, 0, 4],
 [4, 0, 4, 4, 4, 0, 4, 4],
 [0, 4, 4, 4, 0, 4, 4, 4],
 [4, 0, 0, 0, 4, 0, 0, 0],
 [4, 4, 0, 4, 4, 4, 0, 4],
 [4, 0, 4, 4, 4, 0, 4, 4]]
)
    actual = solve_48131b3c(input_grid)
    assert actual == expected



