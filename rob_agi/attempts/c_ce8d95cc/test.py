import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_ce8d95cc.main import solve_ce8d95cc

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_ce8d95cc_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 8, 0, 0, 0, 0, 6, 0, 0],
 [3, 3, 3, 8, 3, 3, 3, 3, 6, 3, 3],
 [0, 0, 0, 8, 0, 0, 0, 0, 6, 0, 0],
 [0, 0, 0, 8, 0, 0, 0, 0, 6, 0, 0],
 [0, 0, 0, 8, 0, 0, 0, 0, 6, 0, 0],
 [0, 0, 0, 8, 0, 0, 0, 0, 6, 0, 0],
 [5, 5, 5, 8, 5, 5, 5, 5, 6, 5, 5],
 [0, 0, 0, 8, 0, 0, 0, 0, 6, 0, 0],
 [0, 0, 0, 8, 0, 0, 0, 0, 6, 0, 0],
 [0, 0, 0, 8, 0, 0, 0, 0, 6, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 8, 0, 6, 0],
 [3, 8, 3, 6, 3],
 [0, 8, 0, 6, 0],
 [5, 8, 5, 6, 5],
 [0, 8, 0, 6, 0]]
)
    actual = solve_ce8d95cc(input_grid)
    assert actual == expected


def test_ce8d95cc_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 1, 0, 0, 8, 0, 3, 0, 0, 0],
 [0, 0, 1, 0, 0, 8, 0, 3, 0, 0, 0],
 [0, 0, 1, 0, 0, 8, 0, 3, 0, 0, 0],
 [2, 2, 1, 2, 2, 8, 2, 3, 2, 2, 2],
 [0, 0, 1, 0, 0, 8, 0, 3, 0, 0, 0],
 [0, 0, 1, 0, 0, 8, 0, 3, 0, 0, 0],
 [0, 0, 1, 0, 0, 8, 0, 3, 0, 0, 0],
 [0, 0, 1, 0, 0, 8, 0, 3, 0, 0, 0],
 [0, 0, 1, 0, 0, 8, 0, 3, 0, 0, 0],
 [5, 5, 1, 5, 5, 8, 5, 3, 5, 5, 5],
 [0, 0, 1, 0, 0, 8, 0, 3, 0, 0, 0],
 [0, 0, 1, 0, 0, 8, 0, 3, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 1, 0, 8, 0, 3, 0],
 [2, 1, 2, 8, 2, 3, 2],
 [0, 1, 0, 8, 0, 3, 0],
 [5, 1, 5, 8, 5, 3, 5],
 [0, 1, 0, 8, 0, 3, 0]]
)
    actual = solve_ce8d95cc(input_grid)
    assert actual == expected


def test_ce8d95cc_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 4, 0, 0, 0, 0, 0, 0],
 [0, 0, 4, 0, 0, 0, 0, 0, 0],
 [0, 0, 4, 0, 0, 0, 0, 0, 0],
 [0, 0, 4, 0, 0, 0, 0, 0, 0],
 [3, 3, 4, 3, 3, 3, 3, 3, 3],
 [0, 0, 4, 0, 0, 0, 0, 0, 0],
 [0, 0, 4, 0, 0, 0, 0, 0, 0],
 [0, 0, 4, 0, 0, 0, 0, 0, 0],
 [0, 0, 4, 0, 0, 0, 0, 0, 0],
 [8, 8, 8, 8, 8, 8, 8, 8, 8],
 [0, 0, 4, 0, 0, 0, 0, 0, 0],
 [0, 0, 4, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 4, 0], [3, 4, 3], [0, 4, 0], [8, 8, 8], [0, 4, 0]]
)
    actual = solve_ce8d95cc(input_grid)
    assert actual == expected


def test_ce8d95cc_example_3():
    input_grid = ColoredGrid(values=
[[0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0],
 [7, 7, 3, 7, 7, 7, 7, 1, 7, 7, 7],
 [0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0],
 [2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2],
 [0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0],
 [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 3, 0, 1, 0],
 [7, 3, 7, 1, 7],
 [0, 3, 0, 1, 0],
 [2, 2, 2, 1, 2],
 [0, 3, 0, 1, 0],
 [8, 8, 8, 8, 8],
 [0, 3, 0, 1, 0]]
)
    actual = solve_ce8d95cc(input_grid)
    assert actual == expected



