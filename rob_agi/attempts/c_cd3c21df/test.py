import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_cd3c21df.main import solve_cd3c21df

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_cd3c21df_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 5, 0, 0, 0, 0, 4, 0, 0, 0],
 [0, 0, 5, 0, 0, 0, 0, 4, 0, 2, 0],
 [0, 0, 5, 0, 0, 0, 0, 4, 0, 2, 0],
 [0, 0, 5, 0, 0, 0, 0, 0, 0, 2, 0],
 [0, 0, 0, 0, 5, 0, 0, 0, 0, 2, 0],
 [0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
 [4, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
 [4, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
 [4, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2], [2], [2], [2]]
)
    actual = solve_cd3c21df(input_grid)
    assert actual == expected


def test_cd3c21df_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 8, 2, 0, 0, 0, 0, 6, 6, 6, 0],
 [0, 0, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 8, 2, 0, 0],
 [6, 6, 6, 0, 0, 0, 0, 0, 2, 8, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[7, 7], [7, 7]]
)
    actual = solve_cd3c21df(input_grid)
    assert actual == expected


def test_cd3c21df_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 6, 0, 0, 0, 0, 2, 2, 2, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 2, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
 [0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 2, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 6, 6, 6, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 1, 2], [1, 1, 1]]
)
    actual = solve_cd3c21df(input_grid)
    assert actual == expected



