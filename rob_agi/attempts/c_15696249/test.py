import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_15696249.main import solve_15696249

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_15696249_example_0():
    input_grid = ColoredGrid(values=
[[4, 4, 6], [3, 3, 3], [6, 6, 4]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [4, 4, 6, 4, 4, 6, 4, 4, 6],
 [3, 3, 3, 3, 3, 3, 3, 3, 3],
 [6, 6, 4, 6, 6, 4, 6, 6, 4],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_15696249(input_grid)
    assert actual == expected


def test_15696249_example_1():
    input_grid = ColoredGrid(values=
[[2, 4, 3], [2, 3, 4], [2, 3, 4]]
    )
    expected = ColoredGrid(values=
[[2, 4, 3, 0, 0, 0, 0, 0, 0],
 [2, 3, 4, 0, 0, 0, 0, 0, 0],
 [2, 3, 4, 0, 0, 0, 0, 0, 0],
 [2, 4, 3, 0, 0, 0, 0, 0, 0],
 [2, 3, 4, 0, 0, 0, 0, 0, 0],
 [2, 3, 4, 0, 0, 0, 0, 0, 0],
 [2, 4, 3, 0, 0, 0, 0, 0, 0],
 [2, 3, 4, 0, 0, 0, 0, 0, 0],
 [2, 3, 4, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_15696249(input_grid)
    assert actual == expected


def test_15696249_example_2():
    input_grid = ColoredGrid(values=
[[1, 1, 1], [6, 2, 2], [2, 2, 6]]
    )
    expected = ColoredGrid(values=
[[1, 1, 1, 1, 1, 1, 1, 1, 1],
 [6, 2, 2, 6, 2, 2, 6, 2, 2],
 [2, 2, 6, 2, 2, 6, 2, 2, 6],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_15696249(input_grid)
    assert actual == expected


def test_15696249_example_3():
    input_grid = ColoredGrid(values=
[[3, 1, 6], [3, 6, 1], [3, 1, 6]]
    )
    expected = ColoredGrid(values=
[[3, 1, 6, 0, 0, 0, 0, 0, 0],
 [3, 6, 1, 0, 0, 0, 0, 0, 0],
 [3, 1, 6, 0, 0, 0, 0, 0, 0],
 [3, 1, 6, 0, 0, 0, 0, 0, 0],
 [3, 6, 1, 0, 0, 0, 0, 0, 0],
 [3, 1, 6, 0, 0, 0, 0, 0, 0],
 [3, 1, 6, 0, 0, 0, 0, 0, 0],
 [3, 6, 1, 0, 0, 0, 0, 0, 0],
 [3, 1, 6, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_15696249(input_grid)
    assert actual == expected



