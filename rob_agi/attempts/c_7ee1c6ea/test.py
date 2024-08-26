import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_7ee1c6ea.main import solve_7ee1c6ea

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_7ee1c6ea_example_0():
    input_grid = ColoredGrid(values=
[[9, 4, 0, 0, 4, 9, 0, 0, 9, 9],
 [4, 9, 9, 4, 9, 9, 0, 0, 9, 0],
 [0, 0, 5, 5, 5, 5, 5, 5, 0, 9],
 [9, 4, 5, 9, 0, 9, 9, 5, 0, 4],
 [4, 4, 5, 0, 0, 4, 0, 5, 4, 4],
 [9, 4, 5, 4, 9, 0, 9, 5, 0, 0],
 [0, 9, 5, 0, 4, 0, 0, 5, 0, 4],
 [0, 4, 5, 5, 5, 5, 5, 5, 4, 4],
 [9, 0, 9, 9, 4, 0, 9, 0, 0, 0],
 [9, 9, 9, 0, 9, 4, 9, 9, 0, 0]]
    )
    expected = ColoredGrid(values=
[[9, 4, 0, 0, 4, 9, 0, 0, 9, 9],
 [4, 9, 9, 4, 9, 9, 0, 0, 9, 0],
 [0, 0, 5, 5, 5, 5, 5, 5, 0, 9],
 [9, 4, 5, 4, 0, 4, 4, 5, 0, 4],
 [4, 4, 5, 0, 0, 9, 0, 5, 4, 4],
 [9, 4, 5, 9, 4, 0, 4, 5, 0, 0],
 [0, 9, 5, 0, 9, 0, 0, 5, 0, 4],
 [0, 4, 5, 5, 5, 5, 5, 5, 4, 4],
 [9, 0, 9, 9, 4, 0, 9, 0, 0, 0],
 [9, 9, 9, 0, 9, 4, 9, 9, 0, 0]]
)
    actual = solve_7ee1c6ea(input_grid)
    assert actual == expected


def test_7ee1c6ea_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 8, 6, 0, 6, 0, 8, 0, 8],
 [8, 5, 5, 5, 5, 5, 5, 5, 5, 0],
 [0, 5, 0, 8, 8, 6, 6, 0, 5, 8],
 [6, 5, 6, 6, 6, 8, 0, 6, 5, 8],
 [0, 5, 6, 6, 8, 6, 0, 6, 5, 8],
 [6, 5, 8, 8, 8, 6, 8, 0, 5, 8],
 [6, 5, 6, 8, 6, 8, 6, 8, 5, 8],
 [0, 5, 6, 0, 6, 8, 8, 8, 5, 8],
 [8, 5, 5, 5, 5, 5, 5, 5, 5, 6],
 [8, 8, 8, 0, 8, 8, 6, 0, 6, 6]]
    )
    expected = ColoredGrid(values=
[[0, 0, 8, 6, 0, 6, 0, 8, 0, 8],
 [8, 5, 5, 5, 5, 5, 5, 5, 5, 0],
 [0, 5, 0, 6, 6, 8, 8, 0, 5, 8],
 [6, 5, 8, 8, 8, 6, 0, 8, 5, 8],
 [0, 5, 8, 8, 6, 8, 0, 8, 5, 8],
 [6, 5, 6, 6, 6, 8, 6, 0, 5, 8],
 [6, 5, 8, 6, 8, 6, 8, 6, 5, 8],
 [0, 5, 8, 0, 8, 6, 6, 6, 5, 8],
 [8, 5, 5, 5, 5, 5, 5, 5, 5, 6],
 [8, 8, 8, 0, 8, 8, 6, 0, 6, 6]]
)
    actual = solve_7ee1c6ea(input_grid)
    assert actual == expected


def test_7ee1c6ea_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 3, 3, 3, 3, 2, 0, 2, 0],
 [3, 5, 5, 5, 5, 5, 5, 5, 5, 3],
 [3, 5, 3, 2, 2, 2, 2, 0, 5, 2],
 [0, 5, 0, 3, 0, 3, 2, 2, 5, 2],
 [3, 5, 2, 0, 2, 3, 2, 2, 5, 3],
 [3, 5, 3, 3, 0, 2, 3, 3, 5, 3],
 [3, 5, 3, 3, 3, 0, 3, 2, 5, 2],
 [0, 5, 3, 0, 3, 3, 3, 0, 5, 3],
 [0, 5, 5, 5, 5, 5, 5, 5, 5, 3],
 [2, 0, 3, 3, 3, 2, 3, 2, 3, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 3, 3, 3, 3, 2, 0, 2, 0],
 [3, 5, 5, 5, 5, 5, 5, 5, 5, 3],
 [3, 5, 2, 3, 3, 3, 3, 0, 5, 2],
 [0, 5, 0, 2, 0, 2, 3, 3, 5, 2],
 [3, 5, 3, 0, 3, 2, 3, 3, 5, 3],
 [3, 5, 2, 2, 0, 3, 2, 2, 5, 3],
 [3, 5, 2, 2, 2, 0, 2, 3, 5, 2],
 [0, 5, 2, 0, 2, 2, 2, 0, 5, 3],
 [0, 5, 5, 5, 5, 5, 5, 5, 5, 3],
 [2, 0, 3, 3, 3, 2, 3, 2, 3, 0]]
)
    actual = solve_7ee1c6ea(input_grid)
    assert actual == expected



