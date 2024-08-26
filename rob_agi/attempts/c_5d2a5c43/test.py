import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_5d2a5c43.main import solve_5d2a5c43

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_5d2a5c43_example_0():
    input_grid = ColoredGrid(values=
[[4, 4, 4, 4, 1, 0, 0, 0, 0],
 [0, 4, 0, 4, 1, 4, 0, 0, 0],
 [4, 0, 0, 0, 1, 0, 4, 0, 0],
 [0, 4, 4, 0, 1, 0, 0, 0, 0],
 [4, 0, 4, 0, 1, 4, 4, 4, 4],
 [0, 4, 4, 4, 1, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 8, 8, 8],
 [8, 8, 0, 8],
 [8, 8, 0, 0],
 [0, 8, 8, 0],
 [8, 8, 8, 8],
 [0, 8, 8, 8]]
)
    actual = solve_5d2a5c43(input_grid)
    assert actual == expected


def test_5d2a5c43_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 4, 4, 1, 0, 0, 4, 4],
 [0, 4, 4, 4, 1, 0, 0, 0, 0],
 [0, 4, 0, 0, 1, 4, 0, 4, 0],
 [0, 4, 4, 4, 1, 4, 4, 0, 4],
 [0, 4, 4, 4, 1, 4, 0, 4, 4],
 [0, 4, 0, 4, 1, 4, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 8, 8],
 [0, 8, 8, 8],
 [8, 8, 8, 0],
 [8, 8, 8, 8],
 [8, 8, 8, 8],
 [8, 8, 0, 8]]
)
    actual = solve_5d2a5c43(input_grid)
    assert actual == expected


def test_5d2a5c43_example_2():
    input_grid = ColoredGrid(values=
[[4, 0, 4, 0, 1, 4, 0, 4, 4],
 [4, 0, 4, 0, 1, 4, 4, 4, 0],
 [4, 4, 0, 4, 1, 4, 0, 4, 0],
 [0, 4, 0, 0, 1, 4, 0, 0, 4],
 [0, 0, 4, 4, 1, 4, 4, 4, 0],
 [4, 4, 0, 4, 1, 4, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 0, 8, 8],
 [8, 8, 8, 0],
 [8, 8, 8, 8],
 [8, 8, 0, 8],
 [8, 8, 8, 8],
 [8, 8, 0, 8]]
)
    actual = solve_5d2a5c43(input_grid)
    assert actual == expected


def test_5d2a5c43_example_3():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 4, 1, 4, 4, 0, 0],
 [0, 0, 4, 4, 1, 0, 4, 0, 0],
 [4, 0, 4, 4, 1, 0, 4, 4, 0],
 [4, 4, 4, 0, 1, 4, 4, 0, 0],
 [4, 0, 4, 4, 1, 4, 0, 0, 4],
 [0, 0, 0, 0, 1, 4, 4, 4, 4]]
    )
    expected = ColoredGrid(values=
[[8, 8, 0, 8],
 [0, 8, 8, 8],
 [8, 8, 8, 8],
 [8, 8, 8, 0],
 [8, 0, 8, 8],
 [8, 8, 8, 8]]
)
    actual = solve_5d2a5c43(input_grid)
    assert actual == expected


def test_5d2a5c43_example_4():
    input_grid = ColoredGrid(values=
[[4, 0, 0, 4, 1, 0, 4, 0, 4],
 [0, 0, 4, 4, 1, 0, 4, 0, 0],
 [4, 0, 4, 4, 1, 4, 0, 4, 0],
 [0, 4, 0, 4, 1, 4, 0, 4, 4],
 [4, 4, 0, 4, 1, 0, 4, 4, 0],
 [0, 4, 4, 4, 1, 0, 4, 0, 4]]
    )
    expected = ColoredGrid(values=
[[8, 8, 0, 8],
 [0, 8, 8, 8],
 [8, 0, 8, 8],
 [8, 8, 8, 8],
 [8, 8, 8, 8],
 [0, 8, 8, 8]]
)
    actual = solve_5d2a5c43(input_grid)
    assert actual == expected



