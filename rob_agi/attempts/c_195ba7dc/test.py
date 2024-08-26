import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_195ba7dc.main import solve_195ba7dc

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_195ba7dc_example_0():
    input_grid = ColoredGrid(values=
[[0, 7, 7, 0, 7, 7, 2, 7, 0, 0, 0, 0, 7],
 [7, 0, 0, 0, 0, 7, 2, 7, 0, 0, 7, 7, 0],
 [7, 0, 7, 7, 0, 7, 2, 7, 0, 0, 7, 0, 0],
 [0, 7, 0, 0, 0, 0, 2, 7, 0, 7, 0, 7, 0],
 [7, 7, 0, 7, 7, 0, 2, 0, 7, 0, 0, 7, 0]]
    )
    expected = ColoredGrid(values=
[[1, 1, 1, 0, 1, 1],
 [1, 0, 0, 1, 1, 1],
 [1, 0, 1, 1, 0, 1],
 [1, 1, 1, 0, 1, 0],
 [1, 1, 0, 1, 1, 0]]
)
    actual = solve_195ba7dc(input_grid)
    assert actual == expected


def test_195ba7dc_example_1():
    input_grid = ColoredGrid(values=
[[0, 7, 7, 7, 0, 7, 2, 7, 7, 0, 7, 0, 7],
 [0, 0, 0, 7, 0, 7, 2, 0, 7, 7, 7, 0, 7],
 [7, 0, 7, 0, 0, 0, 2, 7, 7, 0, 0, 0, 0],
 [7, 7, 7, 0, 0, 0, 2, 7, 7, 0, 0, 7, 7],
 [0, 7, 7, 0, 7, 7, 2, 7, 7, 7, 0, 0, 7]]
    )
    expected = ColoredGrid(values=
[[1, 1, 1, 1, 0, 1],
 [0, 1, 1, 1, 0, 1],
 [1, 1, 1, 0, 0, 0],
 [1, 1, 1, 0, 1, 1],
 [1, 1, 1, 0, 1, 1]]
)
    actual = solve_195ba7dc(input_grid)
    assert actual == expected


def test_195ba7dc_example_2():
    input_grid = ColoredGrid(values=
[[7, 0, 7, 7, 0, 7, 2, 7, 7, 0, 0, 0, 0],
 [7, 0, 0, 7, 0, 0, 2, 0, 0, 0, 7, 0, 0],
 [0, 7, 7, 0, 0, 0, 2, 0, 0, 7, 7, 0, 0],
 [0, 7, 7, 7, 7, 0, 2, 7, 0, 0, 0, 7, 0],
 [7, 0, 7, 0, 7, 7, 2, 7, 7, 7, 7, 7, 7]]
    )
    expected = ColoredGrid(values=
[[1, 1, 1, 1, 0, 1],
 [1, 0, 0, 1, 0, 0],
 [0, 1, 1, 1, 0, 0],
 [1, 1, 1, 1, 1, 0],
 [1, 1, 1, 1, 1, 1]]
)
    actual = solve_195ba7dc(input_grid)
    assert actual == expected


def test_195ba7dc_example_3():
    input_grid = ColoredGrid(values=
[[7, 7, 0, 0, 7, 0, 2, 0, 7, 7, 7, 7, 7],
 [7, 0, 0, 0, 7, 7, 2, 7, 0, 0, 7, 7, 7],
 [0, 7, 0, 0, 7, 0, 2, 0, 0, 0, 0, 0, 0],
 [7, 7, 0, 7, 7, 7, 2, 7, 0, 7, 0, 0, 0],
 [7, 7, 0, 7, 7, 0, 2, 7, 7, 7, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 1, 1, 1, 1, 1],
 [1, 0, 0, 1, 1, 1],
 [0, 1, 0, 0, 1, 0],
 [1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 0]]
)
    actual = solve_195ba7dc(input_grid)
    assert actual == expected



