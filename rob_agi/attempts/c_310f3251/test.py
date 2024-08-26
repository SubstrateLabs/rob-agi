import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_310f3251.main import solve_310f3251

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_310f3251_example_0():
    input_grid = ColoredGrid(values=
[[0, 0], [0, 7]]
    )
    expected = ColoredGrid(values=
[[2, 0, 2, 0, 2, 0],
 [0, 7, 0, 7, 0, 7],
 [2, 0, 2, 0, 2, 0],
 [0, 7, 0, 7, 0, 7],
 [2, 0, 2, 0, 2, 0],
 [0, 7, 0, 7, 0, 7]]
)
    actual = solve_310f3251(input_grid)
    assert actual == expected


def test_310f3251_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0], [0, 0, 6], [6, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 2, 0, 0, 2, 0, 0, 2, 0],
 [0, 0, 6, 0, 0, 6, 0, 0, 6],
 [6, 0, 0, 6, 0, 0, 6, 0, 0],
 [0, 2, 0, 0, 2, 0, 0, 2, 0],
 [0, 0, 6, 0, 0, 6, 0, 0, 6],
 [6, 0, 0, 6, 0, 0, 6, 0, 0],
 [0, 2, 0, 0, 2, 0, 0, 2, 0],
 [0, 0, 6, 0, 0, 6, 0, 0, 6],
 [6, 0, 0, 6, 0, 0, 6, 0, 0]]
)
    actual = solve_310f3251(input_grid)
    assert actual == expected


def test_310f3251_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0],
 [0, 8, 0, 0, 0],
 [0, 8, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0],
 [2, 8, 0, 0, 0, 2, 8, 0, 0, 0, 2, 8, 0, 0, 0],
 [0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0],
 [2, 8, 0, 0, 0, 2, 8, 0, 0, 0, 2, 8, 0, 0, 0],
 [0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0],
 [2, 8, 0, 0, 0, 2, 8, 0, 0, 0, 2, 8, 0, 0, 0],
 [0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_310f3251(input_grid)
    assert actual == expected


def test_310f3251_example_3():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 0, 0], [0, 5, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0],
 [0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5, 0],
 [2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0],
 [0, 5, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0],
 [0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0],
 [0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5, 0],
 [2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0],
 [0, 5, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0],
 [0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0],
 [0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5, 0],
 [2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0],
 [0, 5, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0]]
)
    actual = solve_310f3251(input_grid)
    assert actual == expected


def test_310f3251_example_4():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2],
 [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2],
 [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2],
 [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]]
)
    actual = solve_310f3251(input_grid)
    assert actual == expected



