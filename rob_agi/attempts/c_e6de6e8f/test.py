import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_e6de6e8f.main import solve_e6de6e8f

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_e6de6e8f_example_0():
    input_grid = ColoredGrid(values=
[[2, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2], [2, 2, 0, 2, 2, 0, 2, 0, 2, 2, 0, 2]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 3, 0, 0, 0],
 [0, 0, 0, 2, 2, 0, 0],
 [0, 0, 0, 2, 2, 0, 0],
 [0, 0, 0, 2, 0, 0, 0],
 [0, 0, 0, 2, 0, 0, 0],
 [0, 0, 0, 2, 2, 0, 0],
 [0, 0, 0, 0, 2, 0, 0],
 [0, 0, 0, 0, 2, 0, 0]]
)
    actual = solve_e6de6e8f(input_grid)
    assert actual == expected


def test_e6de6e8f_example_1():
    input_grid = ColoredGrid(values=
[[0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2], [2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 0, 2]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 3, 0, 0, 0],
 [0, 0, 2, 2, 0, 0, 0],
 [0, 0, 2, 2, 0, 0, 0],
 [0, 0, 0, 2, 2, 0, 0],
 [0, 0, 0, 0, 2, 0, 0],
 [0, 0, 0, 0, 2, 0, 0],
 [0, 0, 0, 0, 2, 0, 0],
 [0, 0, 0, 0, 2, 0, 0]]
)
    actual = solve_e6de6e8f(input_grid)
    assert actual == expected


def test_e6de6e8f_example_2():
    input_grid = ColoredGrid(values=
[[2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2], [2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 0, 2]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 3, 0, 0, 0],
 [0, 0, 0, 2, 2, 0, 0],
 [0, 0, 0, 0, 2, 2, 0],
 [0, 0, 0, 0, 0, 2, 2],
 [0, 0, 0, 0, 0, 0, 2],
 [0, 0, 0, 0, 0, 0, 2],
 [0, 0, 0, 0, 0, 0, 2],
 [0, 0, 0, 0, 0, 0, 2]]
)
    actual = solve_e6de6e8f(input_grid)
    assert actual == expected



