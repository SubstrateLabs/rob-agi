import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_0c786b71.main import solve_0c786b71

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_0c786b71_example_0():
    input_grid = ColoredGrid(values=
[[6, 2, 4, 2], [2, 2, 6, 6], [6, 4, 2, 4]]
    )
    expected = ColoredGrid(values=
[[4, 2, 4, 6, 6, 4, 2, 4],
 [6, 6, 2, 2, 2, 2, 6, 6],
 [2, 4, 2, 6, 6, 2, 4, 2],
 [2, 4, 2, 6, 6, 2, 4, 2],
 [6, 6, 2, 2, 2, 2, 6, 6],
 [4, 2, 4, 6, 6, 4, 2, 4]]
)
    actual = solve_0c786b71(input_grid)
    assert actual == expected


def test_0c786b71_example_1():
    input_grid = ColoredGrid(values=
[[5, 5, 9, 9], [9, 5, 5, 5], [5, 7, 5, 7]]
    )
    expected = ColoredGrid(values=
[[7, 5, 7, 5, 5, 7, 5, 7],
 [5, 5, 5, 9, 9, 5, 5, 5],
 [9, 9, 5, 5, 5, 5, 9, 9],
 [9, 9, 5, 5, 5, 5, 9, 9],
 [5, 5, 5, 9, 9, 5, 5, 5],
 [7, 5, 7, 5, 5, 7, 5, 7]]
)
    actual = solve_0c786b71(input_grid)
    assert actual == expected


def test_0c786b71_example_2():
    input_grid = ColoredGrid(values=
[[3, 3, 5, 5], [5, 8, 5, 8], [8, 8, 5, 8]]
    )
    expected = ColoredGrid(values=
[[8, 5, 8, 8, 8, 8, 5, 8],
 [8, 5, 8, 5, 5, 8, 5, 8],
 [5, 5, 3, 3, 3, 3, 5, 5],
 [5, 5, 3, 3, 3, 3, 5, 5],
 [8, 5, 8, 5, 5, 8, 5, 8],
 [8, 5, 8, 8, 8, 8, 5, 8]]
)
    actual = solve_0c786b71(input_grid)
    assert actual == expected



