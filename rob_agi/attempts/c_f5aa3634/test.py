import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_f5aa3634.main import solve_f5aa3634

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_f5aa3634_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 5, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 3, 5, 3, 8, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 7, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 2, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0],
 [0, 0, 3, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 2, 2, 0, 0, 0, 0, 0, 5, 8, 8, 0],
 [0, 0, 3, 2, 0, 0, 0, 0, 3, 5, 3, 8, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 5, 8, 8], [3, 5, 3, 8], [0, 3, 3, 0]]
)
    actual = solve_f5aa3634(input_grid)
    assert actual == expected


def test_f5aa3634_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 2, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 3, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 3, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0],
 [0, 0, 0, 8, 0, 0, 0, 2, 0, 0, 0, 0, 0],
 [0, 0, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0],
 [0, 0, 0, 8, 0, 0, 0, 0, 0, 6, 8, 0, 0],
 [0, 0, 8, 8, 8, 0, 0, 0, 6, 6, 8, 0, 0],
 [0, 0, 5, 5, 5, 0, 0, 0, 0, 4, 4, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 8, 0], [8, 8, 8], [5, 5, 5]]
)
    actual = solve_f5aa3634(input_grid)
    assert actual == expected


def test_f5aa3634_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 2, 8, 0, 0, 0, 0, 0, 0, 0, 5, 9, 0, 0],
 [0, 0, 8, 2, 0, 0, 0, 0, 0, 7, 7, 5, 9, 0, 0],
 [0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 5, 7, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 6, 6, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 3, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 9, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 5, 9, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 7, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 5, 9], [7, 7, 5, 9], [0, 5, 7, 0]]
)
    actual = solve_f5aa3634(input_grid)
    assert actual == expected



