import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_5ffb2104.main import solve_5ffb2104

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_5ffb2104_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 2, 0, 0, 0, 5, 5, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0],
 [0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 6, 6, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 5],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 8],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_5ffb2104(input_grid)
    assert actual == expected


def test_5ffb2104_example_1():
    input_grid = ColoredGrid(values=
[[0, 3, 0, 0, 0, 0],
 [0, 3, 0, 2, 0, 0],
 [0, 0, 0, 2, 0, 0],
 [0, 8, 0, 0, 2, 2],
 [0, 0, 0, 0, 2, 2],
 [6, 6, 6, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 3, 0],
 [0, 0, 0, 0, 3, 2],
 [0, 0, 0, 0, 0, 2],
 [0, 0, 0, 8, 2, 2],
 [0, 0, 0, 0, 2, 2],
 [0, 0, 0, 6, 6, 6]]
)
    actual = solve_5ffb2104(input_grid)
    assert actual == expected


def test_5ffb2104_example_2():
    input_grid = ColoredGrid(values=
[[0, 2, 2, 0, 0, 0],
 [6, 0, 2, 0, 0, 0],
 [6, 0, 0, 0, 0, 0],
 [0, 0, 8, 0, 3, 0],
 [0, 0, 0, 0, 3, 0],
 [8, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 2, 2],
 [0, 0, 0, 0, 6, 2],
 [0, 0, 0, 0, 6, 0],
 [0, 0, 0, 0, 8, 3],
 [0, 0, 0, 0, 0, 3],
 [0, 0, 0, 0, 0, 8]]
)
    actual = solve_5ffb2104(input_grid)
    assert actual == expected



