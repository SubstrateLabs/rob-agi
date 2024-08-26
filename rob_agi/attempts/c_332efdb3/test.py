import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_332efdb3.main import solve_332efdb3

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_332efdb3_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 1, 1, 1, 1, 1, 1],
 [1, 0, 1, 0, 1, 0, 1],
 [1, 1, 1, 1, 1, 1, 1],
 [1, 0, 1, 0, 1, 0, 1],
 [1, 1, 1, 1, 1, 1, 1],
 [1, 0, 1, 0, 1, 0, 1],
 [1, 1, 1, 1, 1, 1, 1]]
)
    actual = solve_332efdb3(input_grid)
    assert actual == expected


def test_332efdb3_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 1, 1, 1, 1],
 [1, 0, 1, 0, 1],
 [1, 1, 1, 1, 1],
 [1, 0, 1, 0, 1],
 [1, 1, 1, 1, 1]]
)
    actual = solve_332efdb3(input_grid)
    assert actual == expected


def test_332efdb3_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 0, 1, 0, 1, 0, 1, 0, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 0, 1, 0, 1, 0, 1, 0, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 0, 1, 0, 1, 0, 1, 0, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 0, 1, 0, 1, 0, 1, 0, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1]]
)
    actual = solve_332efdb3(input_grid)
    assert actual == expected



