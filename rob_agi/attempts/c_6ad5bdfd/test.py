import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_6ad5bdfd.main import solve_6ad5bdfd

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_6ad5bdfd_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 8, 8],
 [3, 0, 0, 4, 0, 0],
 [3, 0, 0, 4, 0, 0],
 [0, 0, 0, 0, 0, 6],
 [1, 1, 0, 0, 0, 6],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 5, 5, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [2, 2, 2, 2, 2, 2]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [3, 0, 0, 4, 8, 8],
 [3, 0, 0, 4, 0, 6],
 [1, 1, 5, 5, 0, 6],
 [2, 2, 2, 2, 2, 2]]
)
    actual = solve_6ad5bdfd(input_grid)
    assert actual == expected


def test_6ad5bdfd_example_1():
    input_grid = ColoredGrid(values=
[[2, 0, 0, 3, 3, 0, 0, 4, 4, 0, 0],
 [2, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0],
 [2, 0, 0, 0, 0, 5, 0, 0, 6, 6, 0],
 [2, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
 [2, 0, 7, 7, 0, 0, 0, 8, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 3, 3, 4, 4, 0, 0, 0, 0, 0, 0],
 [2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [2, 5, 6, 6, 0, 0, 0, 0, 0, 0, 0],
 [2, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
 [2, 7, 7, 8, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_6ad5bdfd(input_grid)
    assert actual == expected


def test_6ad5bdfd_example_2():
    input_grid = ColoredGrid(values=
[[0, 4, 4, 0, 0, 0, 0, 0, 0, 2],
 [0, 0, 0, 5, 5, 0, 0, 6, 0, 2],
 [0, 0, 0, 0, 0, 0, 0, 6, 0, 2],
 [0, 9, 0, 0, 8, 8, 0, 0, 0, 2],
 [0, 9, 0, 0, 0, 0, 0, 0, 0, 2]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 4, 4, 2],
 [0, 0, 0, 0, 0, 0, 5, 5, 6, 2],
 [0, 0, 0, 0, 0, 0, 0, 0, 6, 2],
 [0, 0, 0, 0, 0, 0, 9, 8, 8, 2],
 [0, 0, 0, 0, 0, 0, 9, 0, 0, 2]]
)
    actual = solve_6ad5bdfd(input_grid)
    assert actual == expected



