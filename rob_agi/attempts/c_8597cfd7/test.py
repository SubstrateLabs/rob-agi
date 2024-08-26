import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_8597cfd7.main import solve_8597cfd7

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_8597cfd7_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 2, 0, 0, 0, 4, 0, 0],
 [0, 0, 0, 0, 0, 0, 4, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [5, 5, 5, 5, 5, 5, 5, 5, 5],
 [0, 0, 2, 0, 0, 0, 4, 0, 0],
 [0, 0, 2, 0, 0, 0, 4, 0, 0],
 [0, 0, 0, 0, 0, 0, 4, 0, 0],
 [0, 0, 0, 0, 0, 0, 4, 0, 0]]
    )
    expected = ColoredGrid(values=
[[4, 4], [4, 4]]
)
    actual = solve_8597cfd7(input_grid)
    assert actual == expected


def test_8597cfd7_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 2, 0, 0, 0, 4, 0, 0],
 [0, 0, 0, 0, 0, 0, 4, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [5, 5, 5, 5, 5, 5, 5, 5, 5],
 [0, 0, 2, 0, 0, 0, 4, 0, 0],
 [0, 0, 2, 0, 0, 0, 4, 0, 0],
 [0, 0, 2, 0, 0, 0, 4, 0, 0],
 [0, 0, 2, 0, 0, 0, 4, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 2], [2, 2]]
)
    actual = solve_8597cfd7(input_grid)
    assert actual == expected


def test_8597cfd7_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 2, 0, 0, 0, 4, 0, 0],
 [0, 0, 0, 0, 0, 0, 4, 0, 0],
 [0, 0, 0, 0, 0, 0, 4, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [5, 5, 5, 5, 5, 5, 5, 5, 5],
 [0, 0, 2, 0, 0, 0, 4, 0, 0],
 [0, 0, 2, 0, 0, 0, 4, 0, 0],
 [0, 0, 2, 0, 0, 0, 4, 0, 0],
 [0, 0, 2, 0, 0, 0, 4, 0, 0],
 [0, 0, 0, 0, 0, 0, 4, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 2], [2, 2]]
)
    actual = solve_8597cfd7(input_grid)
    assert actual == expected


def test_8597cfd7_example_3():
    input_grid = ColoredGrid(values=
[[0, 0, 2, 0, 0, 0, 4, 0, 0],
 [0, 0, 2, 0, 0, 0, 4, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [5, 5, 5, 5, 5, 5, 5, 5, 5],
 [0, 0, 2, 0, 0, 0, 4, 0, 0],
 [0, 0, 2, 0, 0, 0, 4, 0, 0],
 [0, 0, 2, 0, 0, 0, 4, 0, 0],
 [0, 0, 2, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 2], [2, 2]]
)
    actual = solve_8597cfd7(input_grid)
    assert actual == expected



