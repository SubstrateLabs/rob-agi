import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_d4b1c2b1.main import solve_d4b1c2b1

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_d4b1c2b1_example_0():
    input_grid = ColoredGrid(values=
[[1, 1, 1], [6, 6, 6], [6, 1, 6]]
    )
    expected = ColoredGrid(values=
[[1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1],
 [6, 6, 6, 6, 6, 6],
 [6, 6, 6, 6, 6, 6],
 [6, 6, 1, 1, 6, 6],
 [6, 6, 1, 1, 6, 6]]
)
    actual = solve_d4b1c2b1(input_grid)
    assert actual == expected


def test_d4b1c2b1_example_1():
    input_grid = ColoredGrid(values=
[[4, 4, 7], [8, 7, 7], [8, 8, 4]]
    )
    expected = ColoredGrid(values=
[[4, 4, 4, 4, 4, 4, 7, 7, 7],
 [4, 4, 4, 4, 4, 4, 7, 7, 7],
 [4, 4, 4, 4, 4, 4, 7, 7, 7],
 [8, 8, 8, 7, 7, 7, 7, 7, 7],
 [8, 8, 8, 7, 7, 7, 7, 7, 7],
 [8, 8, 8, 7, 7, 7, 7, 7, 7],
 [8, 8, 8, 8, 8, 8, 4, 4, 4],
 [8, 8, 8, 8, 8, 8, 4, 4, 4],
 [8, 8, 8, 8, 8, 8, 4, 4, 4]]
)
    actual = solve_d4b1c2b1(input_grid)
    assert actual == expected


def test_d4b1c2b1_example_2():
    input_grid = ColoredGrid(values=
[[4, 2, 8], [2, 2, 5], [8, 5, 4]]
    )
    expected = ColoredGrid(values=
[[4, 4, 4, 4, 2, 2, 2, 2, 8, 8, 8, 8],
 [4, 4, 4, 4, 2, 2, 2, 2, 8, 8, 8, 8],
 [4, 4, 4, 4, 2, 2, 2, 2, 8, 8, 8, 8],
 [4, 4, 4, 4, 2, 2, 2, 2, 8, 8, 8, 8],
 [2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5],
 [2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5],
 [2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5],
 [2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5],
 [8, 8, 8, 8, 5, 5, 5, 5, 4, 4, 4, 4],
 [8, 8, 8, 8, 5, 5, 5, 5, 4, 4, 4, 4],
 [8, 8, 8, 8, 5, 5, 5, 5, 4, 4, 4, 4],
 [8, 8, 8, 8, 5, 5, 5, 5, 4, 4, 4, 4]]
)
    actual = solve_d4b1c2b1(input_grid)
    assert actual == expected


def test_d4b1c2b1_example_3():
    input_grid = ColoredGrid(values=
[[8, 8, 8], [8, 8, 8], [8, 8, 8]]
    )
    expected = ColoredGrid(values=
[[8, 8, 8], [8, 8, 8], [8, 8, 8]]
)
    actual = solve_d4b1c2b1(input_grid)
    assert actual == expected


def test_d4b1c2b1_example_4():
    input_grid = ColoredGrid(values=
[[3, 3, 3], [3, 3, 3], [3, 3, 3]]
    )
    expected = ColoredGrid(values=
[[3, 3, 3], [3, 3, 3], [3, 3, 3]]
)
    actual = solve_d4b1c2b1(input_grid)
    assert actual == expected


def test_d4b1c2b1_example_5():
    input_grid = ColoredGrid(values=
[[3, 6, 6], [3, 6, 6], [3, 3, 3]]
    )
    expected = ColoredGrid(values=
[[3, 3, 6, 6, 6, 6],
 [3, 3, 6, 6, 6, 6],
 [3, 3, 6, 6, 6, 6],
 [3, 3, 6, 6, 6, 6],
 [3, 3, 3, 3, 3, 3],
 [3, 3, 3, 3, 3, 3]]
)
    actual = solve_d4b1c2b1(input_grid)
    assert actual == expected


def test_d4b1c2b1_example_6():
    input_grid = ColoredGrid(values=
[[2, 2, 4], [4, 4, 4], [2, 4, 2]]
    )
    expected = ColoredGrid(values=
[[2, 2, 2, 2, 4, 4],
 [2, 2, 2, 2, 4, 4],
 [4, 4, 4, 4, 4, 4],
 [4, 4, 4, 4, 4, 4],
 [2, 2, 4, 4, 2, 2],
 [2, 2, 4, 4, 2, 2]]
)
    actual = solve_d4b1c2b1(input_grid)
    assert actual == expected



