import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_ea9794b1.main import solve_ea9794b1

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_ea9794b1_example_0():
    input_grid = ColoredGrid(values=
[[4, 4, 4, 4, 4, 3, 3, 0, 3, 3],
 [4, 0, 4, 4, 0, 0, 0, 3, 0, 3],
 [0, 0, 4, 0, 4, 0, 0, 0, 3, 0],
 [4, 4, 4, 0, 0, 3, 0, 0, 3, 3],
 [4, 4, 4, 4, 0, 3, 0, 3, 0, 3],
 [9, 9, 9, 0, 9, 0, 0, 8, 8, 8],
 [9, 9, 0, 0, 9, 8, 0, 0, 0, 0],
 [0, 0, 0, 9, 0, 0, 0, 0, 0, 0],
 [0, 9, 0, 0, 0, 8, 0, 8, 0, 0],
 [0, 0, 0, 0, 9, 0, 8, 0, 8, 0]]
    )
    expected = ColoredGrid(values=
[[3, 3, 9, 3, 3],
 [9, 9, 3, 4, 3],
 [0, 0, 4, 3, 4],
 [3, 9, 8, 3, 3],
 [3, 8, 3, 8, 3]]
)
    actual = solve_ea9794b1(input_grid)
    assert actual == expected


def test_ea9794b1_example_1():
    input_grid = ColoredGrid(values=
[[0, 4, 0, 0, 4, 0, 0, 0, 0, 3],
 [0, 4, 4, 4, 4, 3, 3, 3, 3, 3],
 [0, 4, 0, 0, 0, 0, 3, 3, 3, 0],
 [4, 4, 0, 0, 0, 3, 3, 3, 0, 3],
 [0, 0, 4, 4, 0, 3, 3, 0, 0, 0],
 [9, 0, 9, 0, 9, 0, 0, 8, 8, 0],
 [0, 0, 0, 9, 0, 0, 0, 0, 8, 0],
 [9, 9, 0, 9, 0, 0, 8, 8, 8, 0],
 [0, 0, 9, 9, 9, 0, 0, 0, 0, 0],
 [9, 9, 0, 9, 0, 8, 8, 8, 8, 0]]
    )
    expected = ColoredGrid(values=
[[9, 4, 9, 8, 3],
 [3, 3, 3, 3, 3],
 [9, 3, 3, 3, 0],
 [3, 3, 3, 9, 3],
 [3, 3, 8, 9, 0]]
)
    actual = solve_ea9794b1(input_grid)
    assert actual == expected


def test_ea9794b1_example_2():
    input_grid = ColoredGrid(values=
[[4, 0, 0, 0, 0, 3, 0, 0, 0, 3],
 [0, 0, 4, 4, 4, 3, 3, 3, 3, 3],
 [4, 4, 0, 4, 0, 3, 3, 3, 3, 3],
 [4, 4, 4, 0, 0, 3, 0, 0, 0, 0],
 [0, 0, 4, 0, 4, 3, 3, 0, 0, 0],
 [0, 0, 0, 0, 9, 0, 8, 0, 8, 8],
 [9, 0, 9, 0, 9, 8, 0, 8, 0, 0],
 [0, 0, 9, 0, 0, 8, 0, 8, 8, 0],
 [9, 9, 9, 9, 0, 8, 0, 0, 0, 8],
 [0, 9, 9, 0, 0, 8, 8, 8, 8, 8]]
    )
    expected = ColoredGrid(values=
[[3, 8, 0, 8, 3],
 [3, 3, 3, 3, 3],
 [3, 3, 3, 3, 3],
 [3, 9, 9, 9, 8],
 [3, 3, 9, 8, 8]]
)
    actual = solve_ea9794b1(input_grid)
    assert actual == expected


def test_ea9794b1_example_3():
    input_grid = ColoredGrid(values=
[[0, 4, 4, 4, 0, 0, 0, 0, 3, 3],
 [4, 4, 0, 0, 0, 3, 0, 3, 3, 0],
 [4, 0, 0, 4, 4, 0, 3, 3, 3, 0],
 [0, 0, 4, 0, 4, 3, 0, 0, 3, 0],
 [0, 0, 4, 4, 4, 3, 3, 3, 3, 3],
 [0, 9, 0, 9, 9, 0, 0, 0, 8, 0],
 [9, 0, 0, 9, 9, 0, 8, 8, 0, 8],
 [0, 0, 0, 9, 0, 0, 0, 8, 8, 0],
 [0, 0, 9, 9, 0, 8, 0, 8, 0, 0],
 [9, 9, 0, 9, 0, 0, 8, 0, 8, 8]]
    )
    expected = ColoredGrid(values=
[[0, 9, 4, 3, 3],
 [3, 8, 3, 3, 9],
 [4, 3, 3, 3, 4],
 [3, 0, 9, 3, 4],
 [3, 3, 3, 3, 3]]
)
    actual = solve_ea9794b1(input_grid)
    assert actual == expected


def test_ea9794b1_example_4():
    input_grid = ColoredGrid(values=
[[0, 4, 4, 4, 0, 0, 3, 0, 3, 0],
 [0, 4, 0, 0, 0, 0, 3, 0, 0, 3],
 [0, 0, 0, 0, 0, 0, 0, 3, 3, 3],
 [0, 0, 4, 4, 0, 3, 0, 3, 3, 3],
 [0, 4, 4, 4, 4, 3, 3, 3, 3, 3],
 [9, 0, 9, 9, 0, 0, 0, 0, 0, 0],
 [9, 0, 0, 0, 9, 0, 8, 0, 8, 0],
 [0, 0, 9, 0, 0, 0, 0, 0, 0, 8],
 [0, 0, 0, 9, 0, 8, 8, 0, 0, 0],
 [0, 0, 0, 0, 9, 0, 0, 8, 8, 8]]
    )
    expected = ColoredGrid(values=
[[9, 3, 9, 3, 0],
 [9, 3, 0, 8, 3],
 [0, 0, 3, 3, 3],
 [3, 8, 3, 3, 3],
 [3, 3, 3, 3, 3]]
)
    actual = solve_ea9794b1(input_grid)
    assert actual == expected


def test_ea9794b1_example_5():
    input_grid = ColoredGrid(values=
[[4, 0, 0, 0, 4, 0, 0, 3, 3, 0],
 [4, 0, 0, 0, 0, 3, 3, 3, 3, 0],
 [0, 4, 4, 0, 4, 3, 0, 0, 3, 3],
 [0, 4, 4, 0, 4, 0, 0, 3, 0, 0],
 [0, 0, 0, 0, 4, 0, 0, 0, 0, 3],
 [0, 9, 9, 9, 9, 0, 8, 0, 0, 8],
 [0, 0, 9, 9, 9, 8, 0, 0, 0, 8],
 [9, 9, 9, 0, 0, 8, 8, 0, 8, 0],
 [9, 9, 9, 0, 9, 0, 8, 8, 8, 8],
 [0, 9, 9, 0, 9, 0, 8, 0, 0, 8]]
    )
    expected = ColoredGrid(values=
[[4, 9, 3, 3, 9],
 [3, 3, 3, 3, 9],
 [3, 9, 9, 3, 3],
 [9, 9, 3, 8, 9],
 [0, 9, 9, 0, 3]]
)
    actual = solve_ea9794b1(input_grid)
    assert actual == expected



