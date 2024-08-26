import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_bbb1b8b6.main import solve_bbb1b8b6

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_bbb1b8b6_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 1, 5, 2, 2, 2, 0],
 [1, 0, 0, 0, 5, 0, 2, 2, 2],
 [1, 1, 0, 0, 5, 0, 0, 2, 2],
 [1, 1, 1, 0, 5, 0, 0, 0, 2]]
    )
    expected = ColoredGrid(values=
[[2, 2, 2, 1], [1, 2, 2, 2], [1, 1, 2, 2], [1, 1, 1, 2]]
)
    actual = solve_bbb1b8b6(input_grid)
    assert actual == expected


def test_bbb1b8b6_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 1, 5, 2, 2, 0, 0],
 [1, 0, 0, 0, 5, 2, 2, 0, 0],
 [1, 1, 0, 0, 5, 0, 2, 2, 0],
 [1, 1, 1, 0, 5, 0, 2, 2, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 1], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]
)
    actual = solve_bbb1b8b6(input_grid)
    assert actual == expected


def test_bbb1b8b6_example_2():
    input_grid = ColoredGrid(values=
[[1, 1, 0, 0, 5, 0, 0, 3, 3],
 [1, 0, 0, 1, 5, 0, 3, 3, 0],
 [1, 0, 0, 1, 5, 0, 3, 3, 0],
 [1, 1, 0, 0, 5, 0, 0, 3, 3]]
    )
    expected = ColoredGrid(values=
[[1, 1, 3, 3], [1, 3, 3, 1], [1, 3, 3, 1], [1, 1, 3, 3]]
)
    actual = solve_bbb1b8b6(input_grid)
    assert actual == expected


def test_bbb1b8b6_example_3():
    input_grid = ColoredGrid(values=
[[1, 1, 1, 1, 5, 0, 0, 0, 0],
 [1, 0, 0, 1, 5, 0, 6, 6, 0],
 [1, 0, 0, 1, 5, 0, 6, 6, 0],
 [1, 1, 1, 1, 5, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 1, 1, 1], [1, 6, 6, 1], [1, 6, 6, 1], [1, 1, 1, 1]]
)
    actual = solve_bbb1b8b6(input_grid)
    assert actual == expected


def test_bbb1b8b6_example_4():
    input_grid = ColoredGrid(values=
[[1, 1, 1, 1, 5, 2, 2, 0, 0],
 [1, 0, 0, 1, 5, 2, 2, 0, 0],
 [1, 0, 0, 1, 5, 0, 0, 0, 0],
 [1, 1, 1, 1, 5, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]]
)
    actual = solve_bbb1b8b6(input_grid)
    assert actual == expected


def test_bbb1b8b6_example_5():
    input_grid = ColoredGrid(values=
[[1, 1, 1, 1, 5, 3, 3, 0, 0],
 [1, 0, 0, 1, 5, 3, 3, 0, 0],
 [1, 0, 0, 1, 5, 3, 0, 0, 0],
 [1, 0, 0, 1, 5, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]]
)
    actual = solve_bbb1b8b6(input_grid)
    assert actual == expected


def test_bbb1b8b6_example_6():
    input_grid = ColoredGrid(values=
[[1, 1, 1, 1, 5, 0, 0, 0, 0],
 [1, 0, 0, 0, 5, 0, 7, 7, 7],
 [1, 0, 1, 1, 5, 0, 7, 0, 0],
 [1, 0, 1, 0, 5, 0, 7, 0, 7]]
    )
    expected = ColoredGrid(values=
[[1, 1, 1, 1], [1, 7, 7, 7], [1, 7, 1, 1], [1, 7, 1, 7]]
)
    actual = solve_bbb1b8b6(input_grid)
    assert actual == expected



