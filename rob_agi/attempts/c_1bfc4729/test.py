import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_1bfc4729.main import solve_1bfc4729

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_1bfc4729_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 6, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 7, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 6],
 [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 6],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 6],
 [7, 0, 0, 0, 0, 0, 0, 0, 0, 7],
 [7, 0, 0, 0, 0, 0, 0, 0, 0, 7],
 [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
 [7, 0, 0, 0, 0, 0, 0, 0, 0, 7],
 [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]]
)
    actual = solve_1bfc4729(input_grid)
    assert actual == expected


def test_1bfc4729_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 [4, 0, 0, 0, 0, 0, 0, 0, 0, 4],
 [4, 0, 0, 0, 0, 0, 0, 0, 0, 4],
 [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
 [4, 0, 0, 0, 0, 0, 0, 0, 0, 4],
 [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]
)
    actual = solve_1bfc4729(input_grid)
    assert actual == expected



