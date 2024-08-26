import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_0becf7df.main import solve_0becf7df

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_0becf7df_example_0():
    input_grid = ColoredGrid(values=
[[4, 2, 0, 0, 0, 0, 0, 0, 0, 0],
 [3, 7, 0, 0, 0, 0, 4, 0, 0, 0],
 [0, 0, 0, 0, 0, 3, 4, 4, 0, 0],
 [0, 0, 0, 0, 0, 3, 2, 4, 0, 0],
 [0, 0, 0, 7, 7, 3, 2, 4, 0, 0],
 [0, 0, 0, 7, 3, 3, 2, 0, 0, 0],
 [0, 0, 0, 7, 0, 0, 2, 2, 0, 0],
 [0, 0, 0, 7, 7, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[4, 2, 0, 0, 0, 0, 0, 0, 0, 0],
 [3, 7, 0, 0, 0, 0, 2, 0, 0, 0],
 [0, 0, 0, 0, 0, 7, 2, 2, 0, 0],
 [0, 0, 0, 0, 0, 7, 4, 2, 0, 0],
 [0, 0, 0, 3, 3, 7, 4, 2, 0, 0],
 [0, 0, 0, 3, 7, 7, 4, 0, 0, 0],
 [0, 0, 0, 3, 0, 0, 4, 4, 0, 0],
 [0, 0, 0, 3, 3, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_0becf7df(input_grid)
    assert actual == expected


def test_0becf7df_example_1():
    input_grid = ColoredGrid(values=
[[1, 3, 0, 0, 0, 0, 0, 0, 0, 0],
 [2, 8, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
 [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
 [0, 0, 3, 3, 3, 3, 1, 8, 0, 0],
 [0, 0, 3, 3, 2, 0, 8, 8, 0, 0],
 [0, 0, 0, 0, 2, 0, 8, 8, 0, 0],
 [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 3, 0, 0, 0, 0, 0, 0, 0, 0],
 [2, 8, 0, 0, 0, 0, 3, 0, 0, 0],
 [0, 0, 0, 0, 3, 3, 3, 0, 0, 0],
 [0, 0, 0, 0, 3, 3, 3, 0, 0, 0],
 [0, 0, 1, 1, 1, 1, 3, 2, 0, 0],
 [0, 0, 1, 1, 8, 0, 2, 2, 0, 0],
 [0, 0, 0, 0, 8, 0, 2, 2, 0, 0],
 [0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_0becf7df(input_grid)
    assert actual == expected


def test_0becf7df_example_2():
    input_grid = ColoredGrid(values=
[[9, 4, 0, 0, 0, 0, 0, 0, 0, 0],
 [7, 6, 0, 0, 0, 9, 9, 0, 0, 0],
 [0, 0, 0, 0, 0, 7, 9, 0, 0, 0],
 [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
 [0, 0, 0, 0, 7, 4, 0, 0, 0, 0],
 [0, 0, 0, 6, 6, 7, 0, 0, 0, 0],
 [0, 0, 0, 7, 6, 6, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[9, 4, 0, 0, 0, 0, 0, 0, 0, 0],
 [7, 6, 0, 0, 0, 4, 4, 0, 0, 0],
 [0, 0, 0, 0, 0, 6, 4, 0, 0, 0],
 [0, 0, 0, 0, 0, 9, 0, 0, 0, 0],
 [0, 0, 0, 0, 6, 9, 0, 0, 0, 0],
 [0, 0, 0, 7, 7, 6, 0, 0, 0, 0],
 [0, 0, 0, 6, 7, 7, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_0becf7df(input_grid)
    assert actual == expected



