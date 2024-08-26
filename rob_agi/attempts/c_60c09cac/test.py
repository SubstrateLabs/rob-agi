import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_60c09cac.main import solve_60c09cac

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_60c09cac_example_0():
    input_grid = ColoredGrid(values=
[[0, 3, 0], [0, 7, 7], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 3, 3, 0, 0],
 [0, 0, 3, 3, 0, 0],
 [0, 0, 7, 7, 7, 7],
 [0, 0, 7, 7, 7, 7],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0]]
)
    actual = solve_60c09cac(input_grid)
    assert actual == expected


def test_60c09cac_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 8, 0], [0, 8, 5, 5], [0, 0, 0, 5], [0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 8, 8, 0, 0],
 [0, 0, 0, 0, 8, 8, 0, 0],
 [0, 0, 8, 8, 5, 5, 5, 5],
 [0, 0, 8, 8, 5, 5, 5, 5],
 [0, 0, 0, 0, 0, 0, 5, 5],
 [0, 0, 0, 0, 0, 0, 5, 5],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_60c09cac(input_grid)
    assert actual == expected



