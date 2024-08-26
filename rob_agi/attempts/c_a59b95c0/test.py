import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_a59b95c0.main import solve_a59b95c0

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_a59b95c0_example_0():
    input_grid = ColoredGrid(values=
[[9, 7, 9], [9, 6, 7], [7, 6, 6]]
    )
    expected = ColoredGrid(values=
[[9, 7, 9, 9, 7, 9, 9, 7, 9],
 [9, 6, 7, 9, 6, 7, 9, 6, 7],
 [7, 6, 6, 7, 6, 6, 7, 6, 6],
 [9, 7, 9, 9, 7, 9, 9, 7, 9],
 [9, 6, 7, 9, 6, 7, 9, 6, 7],
 [7, 6, 6, 7, 6, 6, 7, 6, 6],
 [9, 7, 9, 9, 7, 9, 9, 7, 9],
 [9, 6, 7, 9, 6, 7, 9, 6, 7],
 [7, 6, 6, 7, 6, 6, 7, 6, 6]]
)
    actual = solve_a59b95c0(input_grid)
    assert actual == expected


def test_a59b95c0_example_1():
    input_grid = ColoredGrid(values=
[[3, 4, 4], [3, 3, 3], [3, 4, 4]]
    )
    expected = ColoredGrid(values=
[[3, 4, 4, 3, 4, 4],
 [3, 3, 3, 3, 3, 3],
 [3, 4, 4, 3, 4, 4],
 [3, 4, 4, 3, 4, 4],
 [3, 3, 3, 3, 3, 3],
 [3, 4, 4, 3, 4, 4]]
)
    actual = solve_a59b95c0(input_grid)
    assert actual == expected


def test_a59b95c0_example_2():
    input_grid = ColoredGrid(values=
[[8, 2, 1], [1, 8, 3], [2, 1, 3]]
    )
    expected = ColoredGrid(values=
[[8, 2, 1, 8, 2, 1, 8, 2, 1, 8, 2, 1],
 [1, 8, 3, 1, 8, 3, 1, 8, 3, 1, 8, 3],
 [2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3],
 [8, 2, 1, 8, 2, 1, 8, 2, 1, 8, 2, 1],
 [1, 8, 3, 1, 8, 3, 1, 8, 3, 1, 8, 3],
 [2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3],
 [8, 2, 1, 8, 2, 1, 8, 2, 1, 8, 2, 1],
 [1, 8, 3, 1, 8, 3, 1, 8, 3, 1, 8, 3],
 [2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3],
 [8, 2, 1, 8, 2, 1, 8, 2, 1, 8, 2, 1],
 [1, 8, 3, 1, 8, 3, 1, 8, 3, 1, 8, 3],
 [2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3]]
)
    actual = solve_a59b95c0(input_grid)
    assert actual == expected


def test_a59b95c0_example_3():
    input_grid = ColoredGrid(values=
[[7, 7, 7], [7, 2, 2], [7, 7, 2]]
    )
    expected = ColoredGrid(values=
[[7, 7, 7, 7, 7, 7],
 [7, 2, 2, 7, 2, 2],
 [7, 7, 2, 7, 7, 2],
 [7, 7, 7, 7, 7, 7],
 [7, 2, 2, 7, 2, 2],
 [7, 7, 2, 7, 7, 2]]
)
    actual = solve_a59b95c0(input_grid)
    assert actual == expected


def test_a59b95c0_example_4():
    input_grid = ColoredGrid(values=
[[2, 3, 2], [3, 3, 2], [2, 2, 1]]
    )
    expected = ColoredGrid(values=
[[2, 3, 2, 2, 3, 2, 2, 3, 2],
 [3, 3, 2, 3, 3, 2, 3, 3, 2],
 [2, 2, 1, 2, 2, 1, 2, 2, 1],
 [2, 3, 2, 2, 3, 2, 2, 3, 2],
 [3, 3, 2, 3, 3, 2, 3, 3, 2],
 [2, 2, 1, 2, 2, 1, 2, 2, 1],
 [2, 3, 2, 2, 3, 2, 2, 3, 2],
 [3, 3, 2, 3, 3, 2, 3, 3, 2],
 [2, 2, 1, 2, 2, 1, 2, 2, 1]]
)
    actual = solve_a59b95c0(input_grid)
    assert actual == expected



