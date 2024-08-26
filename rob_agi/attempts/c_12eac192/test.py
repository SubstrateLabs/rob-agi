import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_12eac192.main import solve_12eac192

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_12eac192_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 1, 0, 7, 7, 7, 0],
 [8, 8, 0, 0, 5, 5, 0, 0],
 [0, 8, 8, 0, 0, 5, 5, 0],
 [0, 1, 1, 0, 8, 0, 0, 1],
 [0, 7, 0, 1, 8, 0, 0, 0],
 [8, 0, 0, 0, 1, 0, 7, 0],
 [0, 8, 8, 8, 1, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 3, 0, 7, 7, 7, 0],
 [8, 8, 0, 0, 5, 5, 0, 0],
 [0, 8, 8, 0, 0, 5, 5, 0],
 [0, 3, 3, 0, 3, 0, 0, 3],
 [0, 3, 0, 3, 3, 0, 0, 0],
 [3, 0, 0, 0, 3, 0, 3, 0],
 [0, 8, 8, 8, 3, 0, 0, 0]]
)
    actual = solve_12eac192(input_grid)
    assert actual == expected


def test_12eac192_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 1, 8, 1, 1, 1, 0],
 [1, 5, 1, 7, 1, 1, 0, 0],
 [0, 8, 0, 7, 7, 7, 8, 8],
 [0, 8, 8, 0, 0, 0, 8, 0],
 [0, 7, 0, 0, 8, 5, 5, 0],
 [1, 0, 0, 0, 0, 0, 0, 1],
 [1, 0, 8, 7, 7, 8, 0, 0],
 [0, 0, 8, 7, 7, 0, 8, 8],
 [0, 8, 8, 0, 8, 0, 8, 8]]
    )
    expected = ColoredGrid(values=
[[0, 0, 3, 3, 1, 1, 1, 0],
 [3, 3, 3, 7, 1, 1, 0, 0],
 [0, 8, 0, 7, 7, 7, 8, 8],
 [0, 8, 8, 0, 0, 0, 8, 0],
 [0, 3, 0, 0, 3, 3, 3, 0],
 [3, 0, 0, 0, 0, 0, 0, 3],
 [3, 0, 8, 7, 7, 3, 0, 0],
 [0, 0, 8, 7, 7, 0, 8, 8],
 [0, 8, 8, 0, 3, 0, 8, 8]]
)
    actual = solve_12eac192(input_grid)
    assert actual == expected


def test_12eac192_example_2():
    input_grid = ColoredGrid(values=
[[1, 7, 7, 1, 0, 8, 0, 5],
 [1, 7, 7, 1, 1, 0, 1, 0],
 [8, 8, 0, 0, 7, 7, 7, 7],
 [0, 1, 0, 0, 0, 0, 1, 1],
 [5, 0, 8, 0, 1, 0, 1, 1]]
    )
    expected = ColoredGrid(values=
[[3, 7, 7, 1, 0, 3, 0, 3],
 [3, 7, 7, 1, 1, 0, 3, 0],
 [3, 3, 0, 0, 7, 7, 7, 7],
 [0, 3, 0, 0, 0, 0, 1, 1],
 [3, 0, 3, 0, 3, 0, 1, 1]]
)
    actual = solve_12eac192(input_grid)
    assert actual == expected


def test_12eac192_example_3():
    input_grid = ColoredGrid(values=
[[1, 0, 5], [1, 0, 0], [7, 7, 7]]
    )
    expected = ColoredGrid(values=
[[3, 0, 3], [3, 0, 0], [7, 7, 7]]
)
    actual = solve_12eac192(input_grid)
    assert actual == expected



