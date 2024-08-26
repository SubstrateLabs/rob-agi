import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_f3cdc58f.main import solve_f3cdc58f

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_f3cdc58f_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
 [2, 0, 0, 0, 0, 3, 0, 1, 4, 1],
 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
 [1, 4, 0, 0, 0, 0, 0, 0, 0, 1],
 [0, 0, 0, 0, 2, 0, 0, 0, 2, 0],
 [0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 4, 0, 4, 0, 0, 3, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 2, 1, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 4, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 4, 0, 0, 0, 0, 0, 0],
 [1, 2, 0, 4, 0, 0, 0, 0, 0, 0],
 [1, 2, 0, 4, 0, 0, 0, 0, 0, 0],
 [1, 2, 3, 4, 0, 0, 0, 0, 0, 0],
 [1, 2, 3, 4, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_f3cdc58f(input_grid)
    assert actual == expected


def test_f3cdc58f_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 4, 0, 3, 3, 0],
 [0, 1, 3, 0, 0, 0, 3, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 1, 0, 4],
 [3, 0, 0, 0, 2, 0, 0, 0, 2, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
 [0, 3, 0, 0, 0, 4, 3, 2, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 3, 0],
 [0, 0, 4, 0, 0, 4, 0, 1, 0, 1]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 3, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 3, 0, 0, 0, 0, 0, 0, 0],
 [1, 2, 3, 4, 0, 0, 0, 0, 0, 0],
 [1, 2, 3, 4, 0, 0, 0, 0, 0, 0],
 [1, 2, 3, 4, 0, 0, 0, 0, 0, 0],
 [1, 2, 3, 4, 0, 0, 0, 0, 0, 0],
 [1, 2, 3, 4, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_f3cdc58f(input_grid)
    assert actual == expected


def test_f3cdc58f_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 3, 0, 0, 0, 0, 0, 3, 0],
 [0, 1, 0, 0, 2, 0, 0, 4, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 4, 0, 3, 0, 0, 2, 0],
 [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0, 4, 0],
 [0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
 [0, 2, 3, 4, 0, 0, 0, 0, 0, 0],
 [1, 2, 3, 4, 0, 0, 0, 0, 0, 0],
 [1, 2, 3, 4, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_f3cdc58f(input_grid)
    assert actual == expected



