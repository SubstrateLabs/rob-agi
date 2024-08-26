import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_b942fd60.main import solve_b942fd60

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_b942fd60_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
 [0, 0, 3, 0, 0, 0, 0, 0, 7, 0],
 [2, 0, 0, 0, 0, 0, 3, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 7, 0, 8, 0, 0, 6],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 7, 0, 0, 0, 6, 0, 0, 0, 8],
 [0, 0, 6, 0, 8, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 7, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 8, 2, 0, 0, 0, 0],
 [0, 0, 3, 0, 0, 2, 0, 0, 7, 0],
 [2, 2, 2, 2, 2, 2, 3, 0, 0, 0],
 [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
 [0, 0, 0, 0, 7, 2, 8, 0, 0, 6],
 [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
 [0, 7, 0, 0, 0, 6, 0, 0, 0, 8],
 [0, 0, 6, 0, 8, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 7, 0, 0, 0]]
)
    actual = solve_b942fd60(input_grid)
    assert actual == expected


def test_b942fd60_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 3, 7, 0, 0, 0, 0, 0, 0],
 [0, 0, 8, 0, 0, 0, 0, 0, 7, 0, 0, 3],
 [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0],
 [2, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0],
 [0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 7, 0, 0, 7, 0, 0],
 [0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 3, 0, 0, 0, 0, 0, 8, 3, 0, 0, 0],
 [0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 3, 7, 0, 2, 0, 0, 0, 0],
 [0, 0, 8, 2, 2, 2, 2, 2, 7, 0, 0, 3],
 [2, 2, 2, 2, 2, 2, 6, 2, 2, 2, 2, 2],
 [0, 0, 2, 0, 2, 8, 0, 2, 2, 8, 0, 0],
 [2, 2, 2, 8, 2, 0, 0, 2, 2, 0, 0, 0],
 [0, 0, 2, 0, 2, 7, 0, 2, 2, 0, 0, 0],
 [2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 0, 0],
 [0, 0, 7, 0, 2, 0, 0, 2, 2, 0, 0, 0],
 [0, 0, 0, 0, 2, 0, 7, 2, 2, 7, 0, 0],
 [0, 0, 0, 6, 2, 2, 2, 2, 2, 2, 2, 2],
 [0, 3, 0, 0, 2, 0, 0, 8, 3, 0, 0, 0],
 [0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_b942fd60(input_grid)
    assert actual == expected


def test_b942fd60_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 3, 0, 0, 7],
 [0, 0, 0, 0, 0, 0],
 [2, 0, 0, 0, 3, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 8],
 [0, 0, 3, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 3, 2, 0, 7],
 [0, 0, 0, 2, 0, 0],
 [2, 2, 2, 2, 3, 0],
 [0, 0, 0, 2, 0, 0],
 [0, 0, 0, 2, 0, 8],
 [0, 0, 3, 2, 0, 0]]
)
    actual = solve_b942fd60(input_grid)
    assert actual == expected


def test_b942fd60_example_3():
    input_grid = ColoredGrid(values=
[[0, 7, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 3],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [2, 0, 8, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 7, 0, 0, 0, 0, 2, 0],
 [2, 2, 2, 2, 2, 2, 2, 3],
 [0, 2, 0, 0, 0, 0, 2, 0],
 [2, 2, 8, 0, 0, 0, 2, 0],
 [0, 2, 0, 0, 0, 0, 2, 0],
 [0, 2, 0, 0, 0, 0, 2, 0],
 [0, 2, 0, 0, 7, 0, 2, 0],
 [0, 2, 0, 0, 0, 0, 2, 0]]
)
    actual = solve_b942fd60(input_grid)
    assert actual == expected


def test_b942fd60_example_4():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 8, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 2, 0, 0], [2, 2, 2, 2, 8, 0], [0, 0, 0, 2, 0, 0], [0, 0, 0, 2, 0, 0]]
)
    actual = solve_b942fd60(input_grid)
    assert actual == expected


def test_b942fd60_example_5():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 7, 0, 0],
 [6, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [2, 0, 0, 0, 8, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 8],
 [0, 0, 0, 0, 0, 0],
 [7, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 2, 0, 7, 0, 0],
 [6, 2, 2, 2, 2, 2],
 [0, 2, 0, 2, 0, 0],
 [2, 2, 2, 2, 8, 0],
 [0, 2, 0, 2, 0, 0],
 [0, 2, 0, 2, 0, 8],
 [0, 2, 0, 2, 0, 0],
 [7, 2, 0, 2, 0, 0]]
)
    actual = solve_b942fd60(input_grid)
    assert actual == expected



