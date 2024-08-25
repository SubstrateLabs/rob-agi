import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_d2acf2cb.main import solve_d2acf2cb

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_d2acf2cb_example_0():
    input_grid = ColoredGrid(values=
[[0, 6, 0, 0, 0, 6, 6, 0, 0],
 [6, 6, 6, 6, 6, 6, 6, 6, 6],
 [0, 6, 6, 6, 6, 0, 0, 0, 0],
 [6, 6, 0, 0, 0, 6, 6, 0, 0],
 [0, 6, 6, 6, 0, 0, 6, 0, 6],
 [4, 0, 0, 6, 6, 6, 6, 0, 4],
 [0, 6, 6, 6, 0, 6, 6, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 6, 0, 0, 0, 6, 6, 0, 0],
 [6, 6, 6, 6, 6, 6, 6, 6, 6],
 [0, 6, 6, 6, 6, 0, 0, 0, 0],
 [6, 6, 0, 0, 0, 6, 6, 0, 0],
 [0, 6, 6, 6, 0, 0, 6, 0, 6],
 [4, 8, 8, 7, 7, 7, 7, 8, 4],
 [0, 6, 6, 6, 0, 6, 6, 0, 0]]
)
    actual = solve_d2acf2cb(input_grid)
    assert actual == expected


def test_d2acf2cb_example_1():
    input_grid = ColoredGrid(values=
[[0, 6, 0, 6, 6, 0, 6, 0, 6],
 [4, 7, 8, 7, 8, 8, 8, 8, 4],
 [0, 6, 6, 6, 6, 6, 6, 6, 0],
 [0, 0, 6, 0, 6, 6, 0, 0, 6],
 [4, 8, 7, 7, 7, 7, 8, 8, 4],
 [0, 0, 0, 0, 6, 0, 0, 0, 6],
 [6, 0, 6, 0, 6, 0, 0, 6, 0],
 [4, 7, 8, 8, 7, 8, 7, 7, 4],
 [6, 6, 0, 6, 0, 6, 6, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 6, 0, 6, 6, 0, 6, 0, 6],
 [4, 6, 0, 6, 0, 0, 0, 0, 4],
 [0, 6, 6, 6, 6, 6, 6, 6, 0],
 [0, 0, 6, 0, 6, 6, 0, 0, 6],
 [4, 0, 6, 6, 6, 6, 0, 0, 4],
 [0, 0, 0, 0, 6, 0, 0, 0, 6],
 [6, 0, 6, 0, 6, 0, 0, 6, 0],
 [4, 6, 0, 0, 6, 0, 6, 6, 4],
 [6, 6, 0, 6, 0, 6, 6, 0, 0]]
)
    actual = solve_d2acf2cb(input_grid)
    assert actual == expected


def test_d2acf2cb_example_2():
    input_grid = ColoredGrid(values=
[[6, 0, 6, 4, 6, 0, 0, 4, 6],
 [6, 0, 6, 0, 0, 6, 0, 0, 6],
 [0, 6, 6, 0, 0, 0, 0, 6, 0],
 [6, 6, 6, 0, 0, 0, 0, 6, 6],
 [6, 0, 0, 6, 6, 0, 0, 0, 6],
 [6, 6, 6, 4, 0, 6, 6, 4, 0]]
    )
    expected = ColoredGrid(values=
[[6, 0, 6, 4, 6, 0, 0, 4, 6],
 [6, 0, 6, 8, 0, 6, 0, 8, 6],
 [0, 6, 6, 8, 0, 0, 0, 7, 0],
 [6, 6, 6, 8, 0, 0, 0, 7, 6],
 [6, 0, 0, 7, 6, 0, 0, 8, 6],
 [6, 6, 6, 4, 0, 6, 6, 4, 0]]
)
    actual = solve_d2acf2cb(input_grid)
    assert actual == expected



def test_d2acf2cb_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 4, 6, 6, 0, 4, 6, 4, 0],
 [0, 6, 0, 0, 0, 6, 6, 6, 0],
 [0, 0, 0, 6, 0, 0, 6, 6, 6],
 [6, 6, 6, 0, 0, 0, 6, 0, 0],
 [0, 6, 0, 6, 0, 0, 6, 0, 0],
 [0, 6, 6, 0, 6, 6, 0, 6, 6],
 [6, 6, 6, 6, 0, 6, 0, 6, 6],
 [0, 6, 0, 6, 6, 6, 6, 6, 6],
 [6, 0, 0, 0, 6, 0, 0, 6, 0],
 [0, 4, 0, 0, 6, 4, 6, 4, 0]]
    )
    expected = ColoredGrid(values=
[[0, 4, 6, 6, 0, 4, 6, 4, 0],
 [0, 7, 0, 0, 0, 7, 6, 7, 0],
 [0, 8, 0, 6, 0, 8, 6, 7, 6],
 [6, 7, 6, 0, 0, 8, 6, 8, 0],
 [0, 7, 0, 6, 0, 8, 6, 8, 0],
 [0, 7, 6, 0, 6, 7, 0, 7, 6],
 [6, 7, 6, 6, 0, 7, 0, 7, 6],
 [0, 7, 0, 6, 6, 7, 6, 7, 6],
 [6, 8, 0, 0, 6, 8, 0, 7, 0],
 [0, 4, 0, 0, 6, 4, 6, 4, 0]]
    )
    actual = solve_d2acf2cb(input_grid)
    assert actual == expected

