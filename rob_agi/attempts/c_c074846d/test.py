import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_c074846d.main import solve_c074846d

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_c074846d_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 2, 2, 5, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 2, 0, 0, 0],
 [0, 0, 0, 2, 0, 0, 0],
 [0, 3, 3, 5, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_c074846d(input_grid)
    assert actual == expected


def test_c074846d_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 2, 2, 2, 5, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 2, 0, 0],
 [0, 0, 0, 0, 2, 0, 0],
 [0, 0, 0, 0, 2, 0, 0],
 [0, 3, 3, 3, 5, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_c074846d(input_grid)
    assert actual == expected


def test_c074846d_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 2, 0, 0, 0],
 [0, 0, 0, 5, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 3, 0, 0, 0],
 [0, 0, 0, 5, 2, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_c074846d(input_grid)
    assert actual == expected


def test_c074846d_example_3():
    input_grid = ColoredGrid(values=
[[0, 0, 0], [0, 5, 2], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [0, 5, 3], [0, 2, 0]]
)
    actual = solve_c074846d(input_grid)
    assert actual == expected


def test_c074846d_example_4():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 5, 0, 0],
 [0, 0, 0, 0, 2, 0, 0],
 [0, 0, 0, 0, 2, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 2, 2, 5, 0, 0],
 [0, 0, 0, 0, 3, 0, 0],
 [0, 0, 0, 0, 3, 0, 0]]
)
    actual = solve_c074846d(input_grid)
    assert actual == expected



