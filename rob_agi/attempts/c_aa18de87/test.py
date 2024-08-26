import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_aa18de87.main import solve_aa18de87

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_aa18de87_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 3, 0, 0, 0, 0],
 [0, 0, 3, 0, 3, 0, 0, 0],
 [0, 3, 0, 0, 0, 3, 0, 0],
 [3, 0, 0, 0, 0, 0, 3, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 3, 0, 0, 0, 0],
 [0, 0, 3, 2, 3, 0, 0, 0],
 [0, 3, 2, 2, 2, 3, 0, 0],
 [3, 2, 2, 2, 2, 2, 3, 0]]
)
    actual = solve_aa18de87(input_grid)
    assert actual == expected


def test_aa18de87_example_1():
    input_grid = ColoredGrid(values=
[[0, 4, 0, 0, 0, 4, 0, 0], [0, 0, 4, 0, 4, 0, 0, 0], [0, 0, 0, 4, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 4, 2, 2, 2, 4, 0, 0], [0, 0, 4, 2, 4, 0, 0, 0], [0, 0, 0, 4, 0, 0, 0, 0]]
)
    actual = solve_aa18de87(input_grid)
    assert actual == expected


def test_aa18de87_example_2():
    input_grid = ColoredGrid(values=
[[0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0],
 [8, 0, 8, 0, 0, 0, 0, 0, 8, 0, 8, 0],
 [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8],
 [0, 0, 0, 0, 8, 0, 8, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 8, 2, 2, 2, 2, 2, 2, 2, 8, 0, 0],
 [8, 2, 8, 2, 2, 2, 2, 2, 8, 2, 8, 0],
 [0, 0, 0, 8, 2, 2, 2, 8, 2, 2, 2, 8],
 [0, 0, 0, 0, 8, 2, 8, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_aa18de87(input_grid)
    assert actual == expected


def test_aa18de87_example_3():
    input_grid = ColoredGrid(values=
[[1, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 0, 0, 0, 0, 0, 1],
 [0, 0, 1, 0, 0, 0, 1, 0],
 [0, 0, 0, 1, 0, 1, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 2, 2, 2, 2, 2, 1],
 [0, 0, 1, 2, 2, 2, 1, 0],
 [0, 0, 0, 1, 2, 1, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0]]
)
    actual = solve_aa18de87(input_grid)
    assert actual == expected



