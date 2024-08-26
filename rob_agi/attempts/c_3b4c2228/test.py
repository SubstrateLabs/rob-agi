import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_3b4c2228.main import solve_3b4c2228

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_3b4c2228_example_0():
    input_grid = ColoredGrid(values=
[[2, 2, 0, 3, 0, 0, 3],
 [2, 2, 0, 2, 2, 0, 0],
 [0, 0, 0, 2, 2, 0, 0],
 [2, 3, 3, 0, 0, 2, 2],
 [0, 3, 3, 0, 0, 2, 2],
 [0, 0, 0, 0, 3, 3, 0],
 [3, 0, 2, 0, 3, 3, 0]]
    )
    expected = ColoredGrid(values=
[[1, 0, 0], [0, 1, 0], [0, 0, 0]]
)
    actual = solve_3b4c2228(input_grid)
    assert actual == expected


def test_3b4c2228_example_1():
    input_grid = ColoredGrid(values=
[[0, 3, 3, 0, 0],
 [0, 3, 3, 0, 0],
 [0, 0, 0, 0, 0],
 [2, 2, 0, 0, 2],
 [2, 2, 0, 0, 0],
 [0, 0, 0, 2, 2],
 [0, 0, 0, 2, 2]]
    )
    expected = ColoredGrid(values=
[[1, 0, 0], [0, 0, 0], [0, 0, 0]]
)
    actual = solve_3b4c2228(input_grid)
    assert actual == expected


def test_3b4c2228_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 3, 3, 0, 0, 0],
 [2, 0, 3, 3, 0, 3, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [3, 3, 0, 0, 2, 2, 0],
 [3, 3, 0, 0, 2, 2, 0],
 [0, 0, 3, 3, 0, 0, 0],
 [0, 0, 3, 3, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
)
    actual = solve_3b4c2228(input_grid)
    assert actual == expected


def test_3b4c2228_example_3():
    input_grid = ColoredGrid(values=
[[0, 3, 3, 0, 0, 0, 3],
 [0, 3, 3, 0, 0, 0, 0],
 [0, 0, 0, 0, 2, 0, 0],
 [3, 0, 0, 0, 3, 3, 0],
 [0, 0, 3, 0, 3, 3, 0]]
    )
    expected = ColoredGrid(values=
[[1, 0, 0], [0, 1, 0], [0, 0, 0]]
)
    actual = solve_3b4c2228(input_grid)
    assert actual == expected


def test_3b4c2228_example_4():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 2, 2],
 [3, 3, 0, 2, 2],
 [3, 3, 0, 0, 0],
 [0, 0, 2, 2, 0],
 [3, 0, 2, 2, 0]]
    )
    expected = ColoredGrid(values=
[[1, 0, 0], [0, 0, 0], [0, 0, 0]]
)
    actual = solve_3b4c2228(input_grid)
    assert actual == expected



