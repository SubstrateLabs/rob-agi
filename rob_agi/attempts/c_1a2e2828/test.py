import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_1a2e2828.main import solve_1a2e2828

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_1a2e2828_example_0():
    input_grid = ColoredGrid(values=
[[0, 2, 2, 0, 0, 0, 0, 8, 0, 0, 0, 0],
 [0, 2, 2, 0, 0, 0, 0, 8, 0, 0, 0, 0],
 [3, 2, 2, 3, 3, 3, 3, 8, 3, 3, 3, 3],
 [3, 2, 2, 3, 3, 3, 3, 8, 3, 3, 3, 3],
 [0, 2, 2, 0, 0, 0, 0, 8, 0, 0, 0, 0],
 [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
 [0, 2, 2, 0, 0, 0, 0, 8, 0, 0, 0, 0],
 [0, 2, 2, 0, 0, 0, 0, 8, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[6]]
)
    actual = solve_1a2e2828(input_grid)
    assert actual == expected


def test_1a2e2828_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 4, 4, 0, 0, 0, 8, 0, 0],
 [0, 0, 0, 4, 4, 0, 0, 0, 8, 0, 0],
 [3, 3, 3, 4, 4, 3, 3, 3, 8, 3, 3],
 [0, 0, 0, 4, 4, 0, 0, 0, 8, 0, 0],
 [0, 0, 0, 4, 4, 0, 0, 0, 8, 0, 0],
 [6, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6],
 [6, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6],
 [0, 0, 0, 4, 4, 0, 0, 0, 8, 0, 0],
 [0, 0, 0, 4, 4, 0, 0, 0, 8, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8]]
)
    actual = solve_1a2e2828(input_grid)
    assert actual == expected


def test_1a2e2828_example_2():
    input_grid = ColoredGrid(values=
[[0, 2, 2, 0, 6, 0, 0, 8, 8, 0, 0],
 [1, 2, 2, 1, 6, 1, 1, 8, 8, 1, 1],
 [1, 2, 2, 1, 6, 1, 1, 8, 8, 1, 1],
 [1, 2, 2, 1, 6, 1, 1, 8, 8, 1, 1],
 [0, 2, 2, 0, 6, 0, 0, 8, 8, 0, 0],
 [0, 2, 2, 0, 6, 0, 0, 8, 8, 0, 0],
 [4, 4, 4, 4, 6, 4, 4, 4, 4, 4, 4],
 [4, 4, 4, 4, 6, 4, 4, 4, 4, 4, 4],
 [0, 2, 2, 0, 6, 0, 0, 8, 8, 0, 0],
 [0, 2, 2, 0, 6, 0, 0, 8, 8, 0, 0],
 [0, 2, 2, 0, 6, 0, 0, 8, 8, 0, 0]]
    )
    expected = ColoredGrid(values=
[[6]]
)
    actual = solve_1a2e2828(input_grid)
    assert actual == expected


def test_1a2e2828_example_3():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 3, 3, 0, 0, 5, 0, 0, 0],
 [2, 2, 2, 2, 3, 3, 2, 2, 5, 2, 2, 2],
 [0, 0, 0, 0, 3, 3, 0, 0, 5, 0, 0, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [0, 0, 0, 0, 3, 3, 0, 0, 5, 0, 0, 0],
 [4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4],
 [4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4],
 [0, 0, 0, 0, 3, 3, 0, 0, 5, 0, 0, 0],
 [0, 0, 0, 0, 3, 3, 0, 0, 5, 0, 0, 0],
 [0, 0, 0, 0, 3, 3, 0, 0, 5, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1]]
)
    actual = solve_1a2e2828(input_grid)
    assert actual == expected


def test_1a2e2828_example_4():
    input_grid = ColoredGrid(values=
[[0, 1, 0], [3, 3, 3], [0, 1, 0]]
    )
    expected = ColoredGrid(values=
[[3]]
)
    actual = solve_1a2e2828(input_grid)
    assert actual == expected



