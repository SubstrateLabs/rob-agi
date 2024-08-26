import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_e7dd8335.main import solve_e7dd8335

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_e7dd8335_example_0():
    input_grid = ColoredGrid(values=
[[0, 1, 1, 1, 1, 1, 0],
 [0, 1, 0, 1, 0, 1, 0],
 [0, 1, 0, 1, 0, 1, 0],
 [0, 1, 0, 1, 0, 1, 0],
 [0, 1, 0, 1, 0, 1, 0],
 [0, 1, 1, 1, 1, 1, 0],
 [0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 1, 1, 1, 1, 1, 0],
 [0, 1, 0, 1, 0, 1, 0],
 [0, 1, 0, 1, 0, 1, 0],
 [0, 2, 0, 2, 0, 2, 0],
 [0, 2, 0, 2, 0, 2, 0],
 [0, 2, 2, 2, 2, 2, 0],
 [0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_e7dd8335(input_grid)
    assert actual == expected


def test_e7dd8335_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 1, 1, 1, 0, 0, 0, 0],
 [0, 1, 0, 0, 1, 0, 0, 0, 0],
 [0, 1, 0, 0, 1, 0, 0, 0, 0],
 [0, 1, 0, 0, 1, 0, 0, 0, 0],
 [0, 1, 0, 0, 1, 0, 0, 0, 0],
 [0, 1, 1, 1, 1, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 1, 1, 1, 0, 0, 0, 0],
 [0, 1, 0, 0, 1, 0, 0, 0, 0],
 [0, 1, 0, 0, 1, 0, 0, 0, 0],
 [0, 2, 0, 0, 2, 0, 0, 0, 0],
 [0, 2, 0, 0, 2, 0, 0, 0, 0],
 [0, 2, 2, 2, 2, 0, 0, 0, 0]]
)
    actual = solve_e7dd8335(input_grid)
    assert actual == expected


def test_e7dd8335_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 1, 1, 1, 1, 1, 0, 0, 0],
 [0, 0, 1, 0, 1, 0, 0, 0, 0],
 [0, 0, 1, 0, 1, 0, 0, 0, 0],
 [0, 0, 1, 0, 1, 0, 0, 0, 0],
 [0, 0, 1, 0, 1, 0, 0, 0, 0],
 [0, 1, 1, 1, 1, 1, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 1, 1, 1, 1, 1, 0, 0, 0],
 [0, 0, 1, 0, 1, 0, 0, 0, 0],
 [0, 0, 1, 0, 1, 0, 0, 0, 0],
 [0, 0, 2, 0, 2, 0, 0, 0, 0],
 [0, 0, 2, 0, 2, 0, 0, 0, 0],
 [0, 2, 2, 2, 2, 2, 0, 0, 0],
 [0, 0, 0, 2, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_e7dd8335(input_grid)
    assert actual == expected



