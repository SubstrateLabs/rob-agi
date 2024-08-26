import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_73182012.main import solve_73182012

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_73182012_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
 [0, 2, 3, 1, 1, 3, 2, 0, 0, 0, 0, 0],
 [2, 2, 1, 0, 0, 1, 2, 2, 0, 0, 0, 0],
 [2, 2, 1, 0, 0, 1, 2, 2, 0, 0, 0, 0],
 [0, 2, 3, 1, 1, 3, 2, 0, 0, 0, 0, 0],
 [0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 2], [0, 0, 2, 2], [0, 2, 3, 1], [2, 2, 1, 0]]
)
    actual = solve_73182012(input_grid)
    assert actual == expected


def test_73182012_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
 [0, 0, 0, 5, 5, 2, 2, 5, 5, 0, 0, 0],
 [0, 0, 0, 5, 3, 3, 3, 3, 5, 0, 0, 0],
 [0, 0, 2, 2, 3, 1, 1, 3, 2, 2, 0, 0],
 [0, 0, 2, 2, 3, 1, 1, 3, 2, 2, 0, 0],
 [0, 0, 0, 5, 3, 3, 3, 3, 5, 0, 0, 0],
 [0, 0, 0, 5, 5, 2, 2, 5, 5, 0, 0, 0],
 [0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 2], [0, 5, 5, 2], [0, 5, 3, 3], [2, 2, 3, 1]]
)
    actual = solve_73182012(input_grid)
    assert actual == expected


def test_73182012_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 7, 7, 0, 0, 7, 7, 0],
 [0, 0, 0, 0, 7, 2, 2, 3, 3, 2, 2, 7],
 [0, 0, 0, 0, 7, 2, 8, 8, 8, 8, 2, 7],
 [0, 0, 0, 0, 0, 3, 8, 0, 0, 8, 3, 0],
 [0, 0, 0, 0, 0, 3, 8, 0, 0, 8, 3, 0],
 [0, 0, 0, 0, 7, 2, 8, 8, 8, 8, 2, 7],
 [0, 0, 0, 0, 7, 2, 2, 3, 3, 2, 2, 7],
 [0, 0, 0, 0, 0, 7, 7, 0, 0, 7, 7, 0]]
    )
    expected = ColoredGrid(values=
[[0, 7, 7, 0], [7, 2, 2, 3], [7, 2, 8, 8], [0, 3, 8, 0]]
)
    actual = solve_73182012(input_grid)
    assert actual == expected



