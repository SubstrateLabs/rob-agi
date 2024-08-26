import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_5289ad53.main import solve_5289ad53

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_5289ad53_example_0():
    input_grid = ColoredGrid(values=
[[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 8, 8, 8],
 [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [8, 8, 3, 3, 3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 3, 3, 3, 8],
 [8, 8, 8, 8, 8, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]]
    )
    expected = ColoredGrid(values=
[[3, 3, 3], [3, 2, 0]]
)
    actual = solve_5289ad53(input_grid)
    assert actual == expected


def test_5289ad53_example_1():
    input_grid = ColoredGrid(values=
[[5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
 [3, 3, 3, 3, 5, 5, 5, 5, 5, 5],
 [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
 [5, 2, 2, 5, 5, 5, 5, 5, 5, 5],
 [5, 5, 5, 5, 5, 5, 2, 2, 2, 5],
 [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
 [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
 [5, 5, 5, 5, 3, 3, 3, 3, 5, 5],
 [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
 [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]]
    )
    expected = ColoredGrid(values=
[[3, 3, 2], [2, 0, 0]]
)
    actual = solve_5289ad53(input_grid)
    assert actual == expected


def test_5289ad53_example_2():
    input_grid = ColoredGrid(values=
[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1],
 [1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    )
    expected = ColoredGrid(values=
[[3, 3, 3], [2, 2, 0]]
)
    actual = solve_5289ad53(input_grid)
    assert actual == expected


def test_5289ad53_example_3():
    input_grid = ColoredGrid(values=
[[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [8, 8, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [8, 8, 8, 8, 8, 8, 8, 3, 3, 3, 3, 3, 8, 8, 8],
 [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [8, 8, 3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8],
 [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [8, 8, 8, 8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 2, 8],
 [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [8, 2, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [8, 8, 8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 8, 8, 8],
 [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]]
    )
    expected = ColoredGrid(values=
[[3, 3, 2], [2, 2, 2]]
)
    actual = solve_5289ad53(input_grid)
    assert actual == expected



