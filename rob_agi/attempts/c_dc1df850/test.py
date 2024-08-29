import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_dc1df850.main import solve_dc1df850

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_dc1df850_example_0():
    input_grid = ColoredGrid(values=
[[2, 0, 0, 0, 0],
 [0, 0, 0, 2, 0],
 [0, 0, 0, 0, 0],
 [0, 6, 0, 0, 0],
 [0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 1, 1, 1, 1],
 [1, 1, 1, 2, 1],
 [0, 0, 1, 1, 1],
 [0, 6, 0, 0, 0],
 [0, 0, 0, 0, 0]]
)
    actual = solve_dc1df850(input_grid)
    assert actual == expected


def test_dc1df850_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 2],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 3, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 8, 0],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 2, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 1, 2],
 [0, 0, 0, 0, 0, 0, 1, 1],
 [0, 0, 0, 3, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 8, 0],
 [0, 1, 1, 1, 0, 0, 0, 0],
 [0, 1, 2, 1, 0, 0, 0, 0],
 [0, 1, 1, 1, 0, 0, 0, 0]]
)
    actual = solve_dc1df850(input_grid)
    assert actual == expected


def test_dc1df850_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 1, 1, 0, 0], [1, 2, 1, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0]]
)
    actual = solve_dc1df850(input_grid)
    assert actual == expected



