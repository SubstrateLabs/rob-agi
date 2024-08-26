import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_d5c634a2.main import solve_d5c634a2

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_d5c634a2_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
 [2, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
 [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0],
 [0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[3, 0, 3, 1, 0, 1], [0, 0, 0, 0, 0, 0], [3, 0, 0, 1, 0, 1]]
)
    actual = solve_d5c634a2(input_grid)
    assert actual == expected


def test_d5c634a2_example_1():
    input_grid = ColoredGrid(values=
[[2, 2, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0], [2, 2, 2, 0]]
    )
    expected = ColoredGrid(values=
[[3, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
)
    actual = solve_d5c634a2(input_grid)
    assert actual == expected


def test_d5c634a2_example_2():
    input_grid = ColoredGrid(values=
[[2, 2, 2, 0, 0],
 [0, 2, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 2, 2, 2],
 [0, 2, 0, 2, 0],
 [2, 2, 2, 0, 0]]
    )
    expected = ColoredGrid(values=
[[3, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
)
    actual = solve_d5c634a2(input_grid)
    assert actual == expected


def test_d5c634a2_example_3():
    input_grid = ColoredGrid(values=
[[0, 2, 0, 0, 2, 2, 2],
 [2, 2, 2, 0, 0, 2, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 2, 2, 2],
 [0, 0, 2, 0, 0, 2, 0],
 [0, 2, 2, 2, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[3, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0], [3, 0, 0, 1, 0, 0]]
)
    actual = solve_d5c634a2(input_grid)
    assert actual == expected


def test_d5c634a2_example_4():
    input_grid = ColoredGrid(values=
[[0, 2, 2, 2, 0, 0, 0],
 [0, 0, 2, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 2, 0],
 [2, 2, 2, 0, 2, 2, 2],
 [0, 2, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 2, 2, 2, 0, 0],
 [0, 0, 0, 2, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[3, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
)
    actual = solve_d5c634a2(input_grid)
    assert actual == expected


def test_d5c634a2_example_5():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0],
 [2, 2, 2, 0, 0, 0, 0],
 [0, 2, 0, 0, 0, 2, 0],
 [0, 0, 0, 0, 2, 2, 2],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 2, 0, 0, 0, 0],
 [0, 2, 2, 2, 0, 2, 0],
 [0, 0, 0, 0, 2, 2, 2]]
    )
    expected = ColoredGrid(values=
[[3, 0, 3, 1, 0, 0], [0, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0]]
)
    actual = solve_d5c634a2(input_grid)
    assert actual == expected


def test_d5c634a2_example_6():
    input_grid = ColoredGrid(values=
[[0, 2, 0, 0, 0, 0, 0],
 [2, 2, 2, 0, 0, 2, 0],
 [0, 0, 0, 0, 2, 2, 2],
 [0, 0, 2, 0, 0, 0, 0],
 [0, 2, 2, 2, 0, 0, 0],
 [0, 0, 0, 0, 2, 2, 2],
 [0, 2, 0, 0, 0, 2, 0],
 [2, 2, 2, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[3, 0, 3, 1, 0, 0], [0, 0, 0, 0, 0, 0], [3, 0, 3, 0, 0, 0]]
)
    actual = solve_d5c634a2(input_grid)
    assert actual == expected



