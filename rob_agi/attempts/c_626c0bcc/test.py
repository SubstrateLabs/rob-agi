import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_626c0bcc.main import solve_626c0bcc

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_626c0bcc_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 8, 8, 0, 0, 0],
 [8, 8, 8, 8, 8, 0, 0],
 [0, 8, 8, 0, 8, 8, 0],
 [0, 8, 8, 8, 8, 0, 0],
 [0, 0, 0, 8, 8, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 1, 1, 0, 0, 0],
 [3, 3, 1, 1, 4, 0, 0],
 [0, 3, 2, 0, 4, 4, 0],
 [0, 2, 2, 1, 1, 0, 0],
 [0, 0, 0, 1, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_626c0bcc(input_grid)
    assert actual == expected


def test_626c0bcc_example_1():
    input_grid = ColoredGrid(values=
[[0, 8, 0, 0, 8, 0, 0],
 [8, 8, 0, 0, 8, 8, 0],
 [0, 8, 8, 0, 8, 8, 0],
 [0, 8, 8, 0, 8, 8, 0],
 [0, 0, 8, 8, 0, 0, 0],
 [0, 0, 0, 8, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 2, 0, 0, 4, 0, 0],
 [2, 2, 0, 0, 4, 4, 0],
 [0, 1, 1, 0, 1, 1, 0],
 [0, 1, 1, 0, 1, 1, 0],
 [0, 0, 3, 3, 0, 0, 0],
 [0, 0, 0, 3, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_626c0bcc(input_grid)
    assert actual == expected


def test_626c0bcc_example_2():
    input_grid = ColoredGrid(values=
[[8, 8, 8, 0, 0, 0, 0],
 [8, 8, 8, 8, 0, 0, 0],
 [8, 8, 0, 8, 0, 0, 0],
 [0, 8, 8, 8, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 1, 4, 0, 0, 0, 0],
 [1, 1, 4, 4, 0, 0, 0],
 [3, 3, 0, 2, 0, 0, 0],
 [0, 3, 2, 2, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_626c0bcc(input_grid)
    assert actual == expected



