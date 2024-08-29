import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_f5b8619d.main import solve_f5b8619d

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_f5b8619d_example_0():
    input_grid = ColoredGrid(values=
[[2, 0, 0], [0, 0, 0], [0, 0, 2]]
    )
    expected = ColoredGrid(values=
[[2, 0, 8, 2, 0, 8],
 [8, 0, 8, 8, 0, 8],
 [8, 0, 2, 8, 0, 2],
 [2, 0, 8, 2, 0, 8],
 [8, 0, 8, 8, 0, 8],
 [8, 0, 2, 8, 0, 2]]
)
    actual = solve_f5b8619d(input_grid)
    assert actual == expected


def test_f5b8619d_example_1():
    input_grid = ColoredGrid(values=
[[0, 5, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [5, 0, 0, 0, 0, 5],
 [0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 5, 0, 0, 0, 8, 8, 5, 0, 0, 0, 8],
 [8, 8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 8],
 [8, 8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 8],
 [8, 8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 8],
 [5, 8, 0, 0, 0, 5, 5, 8, 0, 0, 0, 5],
 [8, 8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 8],
 [8, 5, 0, 0, 0, 8, 8, 5, 0, 0, 0, 8],
 [8, 8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 8],
 [8, 8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 8],
 [8, 8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 8],
 [5, 8, 0, 0, 0, 5, 5, 8, 0, 0, 0, 5],
 [8, 8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 8]]
)
    actual = solve_f5b8619d(input_grid)
    assert actual == expected


def test_f5b8619d_example_2():
    input_grid = ColoredGrid(values=
[[0, 4], [0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 4, 0, 4], [0, 8, 0, 8], [0, 4, 0, 4], [0, 8, 0, 8]]
)
    actual = solve_f5b8619d(input_grid)
    assert actual == expected



