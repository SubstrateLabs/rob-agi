import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_00576224.main import solve_00576224

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_00576224_example_0():
    input_grid = ColoredGrid(values=
[[8, 6], [6, 4]]
    )
    expected = ColoredGrid(values=
[[8, 6, 8, 6, 8, 6],
 [6, 4, 6, 4, 6, 4],
 [6, 8, 6, 8, 6, 8],
 [4, 6, 4, 6, 4, 6],
 [8, 6, 8, 6, 8, 6],
 [6, 4, 6, 4, 6, 4]]
)
    actual = solve_00576224(input_grid)
    assert actual == expected


def test_00576224_example_1():
    input_grid = ColoredGrid(values=
[[7, 9], [4, 3]]
    )
    expected = ColoredGrid(values=
[[7, 9, 7, 9, 7, 9],
 [4, 3, 4, 3, 4, 3],
 [9, 7, 9, 7, 9, 7],
 [3, 4, 3, 4, 3, 4],
 [7, 9, 7, 9, 7, 9],
 [4, 3, 4, 3, 4, 3]]
)
    actual = solve_00576224(input_grid)
    assert actual == expected



