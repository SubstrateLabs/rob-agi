import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_e99362f0.main import solve_e99362f0

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_e99362f0_example_0():
    input_grid = ColoredGrid(values=
[[7, 0, 0, 0, 4, 0, 0, 9, 0],
 [7, 7, 0, 0, 4, 9, 9, 0, 9],
 [0, 0, 0, 0, 4, 0, 9, 9, 0],
 [0, 0, 7, 0, 4, 0, 0, 0, 0],
 [7, 0, 7, 7, 4, 9, 0, 0, 0],
 [4, 4, 4, 4, 4, 4, 4, 4, 4],
 [2, 0, 2, 0, 4, 0, 0, 0, 0],
 [2, 0, 0, 2, 4, 0, 0, 8, 8],
 [2, 0, 0, 2, 4, 8, 0, 0, 8],
 [0, 0, 0, 2, 4, 0, 8, 0, 0],
 [0, 0, 0, 0, 4, 0, 0, 8, 8]]
    )
    expected = ColoredGrid(values=
[[7, 0, 9, 0], [7, 7, 8, 8], [8, 9, 9, 8], [0, 8, 7, 2], [7, 0, 8, 8]]
)
    actual = solve_e99362f0(input_grid)
    assert actual == expected


def test_e99362f0_example_1():
    input_grid = ColoredGrid(values=
[[0, 7, 7, 0, 4, 0, 0, 0, 0],
 [0, 0, 0, 0, 4, 0, 9, 0, 9],
 [0, 7, 7, 0, 4, 9, 9, 0, 9],
 [7, 0, 7, 7, 4, 0, 0, 0, 9],
 [7, 0, 7, 7, 4, 9, 0, 0, 9],
 [4, 4, 4, 4, 4, 4, 4, 4, 4],
 [0, 0, 2, 2, 4, 8, 8, 8, 0],
 [0, 2, 0, 2, 4, 0, 0, 0, 8],
 [2, 2, 2, 2, 4, 0, 0, 8, 8],
 [0, 0, 2, 2, 4, 8, 0, 0, 0],
 [0, 0, 2, 0, 4, 0, 8, 8, 0]]
    )
    expected = ColoredGrid(values=
[[8, 8, 8, 2], [0, 9, 0, 8], [9, 7, 8, 8], [8, 0, 7, 7], [7, 8, 8, 7]]
)
    actual = solve_e99362f0(input_grid)
    assert actual == expected


def test_e99362f0_example_2():
    input_grid = ColoredGrid(values=
[[7, 7, 7, 0, 4, 9, 0, 0, 0],
 [7, 7, 7, 7, 4, 0, 9, 0, 9],
 [7, 7, 7, 7, 4, 0, 0, 9, 0],
 [0, 7, 0, 7, 4, 9, 9, 9, 9],
 [7, 7, 0, 7, 4, 9, 0, 0, 9],
 [4, 4, 4, 4, 4, 4, 4, 4, 4],
 [0, 2, 0, 2, 4, 0, 0, 0, 8],
 [2, 2, 2, 0, 4, 0, 8, 0, 0],
 [2, 0, 2, 2, 4, 0, 0, 0, 8],
 [0, 0, 2, 2, 4, 0, 8, 0, 0],
 [0, 2, 2, 0, 4, 8, 8, 0, 0]]
    )
    expected = ColoredGrid(values=
[[7, 7, 7, 8], [7, 8, 7, 7], [7, 7, 7, 8], [9, 8, 9, 7], [8, 8, 2, 7]]
)
    actual = solve_e99362f0(input_grid)
    assert actual == expected


def test_e99362f0_example_3():
    input_grid = ColoredGrid(values=
[[0, 7, 0, 0, 4, 9, 0, 9, 0],
 [7, 7, 0, 0, 4, 9, 0, 0, 0],
 [0, 0, 0, 0, 4, 9, 0, 9, 9],
 [0, 7, 7, 7, 4, 0, 0, 0, 0],
 [0, 0, 7, 7, 4, 0, 0, 9, 9],
 [4, 4, 4, 4, 4, 4, 4, 4, 4],
 [0, 2, 2, 0, 4, 8, 8, 0, 0],
 [2, 2, 0, 2, 4, 8, 0, 8, 8],
 [2, 0, 2, 2, 4, 0, 8, 0, 8],
 [2, 0, 2, 2, 4, 0, 8, 8, 0],
 [2, 0, 0, 0, 4, 0, 0, 8, 0]]
    )
    expected = ColoredGrid(values=
[[8, 8, 9, 0], [8, 7, 8, 8], [9, 8, 9, 8], [2, 8, 8, 7], [2, 0, 8, 7]]
)
    actual = solve_e99362f0(input_grid)
    assert actual == expected


def test_e99362f0_example_4():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 4, 0, 9, 0, 0],
 [7, 0, 7, 7, 4, 9, 9, 9, 9],
 [7, 0, 7, 7, 4, 9, 9, 0, 0],
 [7, 7, 0, 0, 4, 0, 0, 9, 0],
 [7, 0, 0, 7, 4, 9, 9, 9, 0],
 [4, 4, 4, 4, 4, 4, 4, 4, 4],
 [0, 2, 2, 2, 4, 8, 0, 0, 0],
 [2, 2, 2, 2, 4, 8, 8, 8, 8],
 [2, 0, 0, 2, 4, 8, 8, 8, 0],
 [2, 2, 0, 0, 4, 0, 8, 8, 8],
 [2, 2, 2, 0, 4, 0, 8, 8, 0]]
    )
    expected = ColoredGrid(values=
[[8, 9, 2, 2], [8, 8, 8, 8], [8, 8, 8, 7], [7, 8, 8, 8], [7, 8, 8, 7]]
)
    actual = solve_e99362f0(input_grid)
    assert actual == expected


def test_e99362f0_example_5():
    input_grid = ColoredGrid(values=
[[7, 0, 7, 7, 4, 0, 9, 9, 9],
 [0, 7, 7, 0, 4, 9, 9, 9, 0],
 [0, 0, 0, 0, 4, 9, 0, 0, 0],
 [7, 0, 0, 7, 4, 9, 9, 9, 0],
 [7, 0, 7, 7, 4, 9, 0, 9, 0],
 [4, 4, 4, 4, 4, 4, 4, 4, 4],
 [0, 2, 0, 0, 4, 0, 0, 8, 0],
 [2, 0, 2, 2, 4, 8, 0, 8, 8],
 [0, 2, 0, 0, 4, 0, 0, 8, 8],
 [2, 0, 2, 2, 4, 8, 0, 0, 8],
 [2, 2, 2, 0, 4, 8, 8, 0, 0]]
    )
    expected = ColoredGrid(values=
[[7, 9, 8, 7], [8, 7, 8, 8], [9, 2, 8, 8], [8, 9, 9, 8], [8, 8, 7, 7]]
)
    actual = solve_e99362f0(input_grid)
    assert actual == expected



