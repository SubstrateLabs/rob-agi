import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_32e9702f.main import solve_32e9702f

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_32e9702f_example_0():
    input_grid = ColoredGrid(values=
[[4, 4, 4], [0, 0, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[4, 4, 5], [5, 5, 5], [5, 5, 5]]
)
    actual = solve_32e9702f(input_grid)
    assert actual == expected


def test_32e9702f_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 3, 3, 3, 3, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 3, 3, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[5, 5, 5, 5, 5, 5, 5, 5],
 [5, 3, 3, 3, 3, 5, 5, 5],
 [5, 5, 5, 5, 5, 5, 5, 5],
 [5, 5, 5, 5, 5, 5, 5, 5],
 [3, 3, 5, 5, 5, 5, 5, 5],
 [5, 5, 5, 5, 5, 5, 5, 5],
 [5, 5, 5, 5, 5, 5, 5, 5],
 [5, 5, 5, 5, 5, 5, 5, 5]]
)
    actual = solve_32e9702f(input_grid)
    assert actual == expected


def test_32e9702f_example_2():
    input_grid = ColoredGrid(values=
[[7, 7, 7, 7, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 7, 7, 7, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 7, 7, 7, 7, 7, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[7, 7, 7, 5, 5, 5, 5],
 [5, 5, 5, 5, 5, 5, 5],
 [5, 7, 7, 7, 5, 5, 5],
 [5, 5, 5, 5, 5, 5, 5],
 [7, 7, 7, 7, 7, 5, 5],
 [5, 5, 5, 5, 5, 5, 5],
 [5, 5, 5, 5, 5, 5, 5]]
)
    actual = solve_32e9702f(input_grid)
    assert actual == expected



