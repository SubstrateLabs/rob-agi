import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_31d5ba1a.main import solve_31d5ba1a

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_31d5ba1a_example_0():
    input_grid = ColoredGrid(values=
[[9, 9, 0, 9, 0],
 [9, 0, 0, 9, 0],
 [0, 9, 9, 9, 9],
 [4, 0, 0, 4, 0],
 [4, 4, 0, 4, 4],
 [4, 4, 4, 0, 4]]
    )
    expected = ColoredGrid(values=
[[0, 6, 0, 0, 0], [0, 6, 0, 0, 6], [6, 0, 0, 6, 0]]
)
    actual = solve_31d5ba1a(input_grid)
    assert actual == expected


def test_31d5ba1a_example_1():
    input_grid = ColoredGrid(values=
[[9, 0, 0, 9, 9],
 [0, 0, 0, 0, 0],
 [0, 0, 9, 0, 9],
 [0, 0, 4, 4, 0],
 [4, 4, 4, 0, 0],
 [4, 0, 4, 0, 4]]
    )
    expected = ColoredGrid(values=
[[6, 0, 6, 0, 6], [6, 6, 6, 0, 0], [6, 0, 0, 0, 0]]
)
    actual = solve_31d5ba1a(input_grid)
    assert actual == expected


def test_31d5ba1a_example_2():
    input_grid = ColoredGrid(values=
[[0, 9, 0, 0, 0],
 [0, 9, 9, 0, 9],
 [9, 0, 0, 0, 9],
 [4, 4, 0, 4, 0],
 [0, 4, 4, 4, 0],
 [4, 4, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[6, 0, 0, 6, 0], [0, 0, 0, 6, 6], [0, 6, 0, 0, 6]]
)
    actual = solve_31d5ba1a(input_grid)
    assert actual == expected


def test_31d5ba1a_example_3():
    input_grid = ColoredGrid(values=
[[0, 0, 9, 9, 0],
 [9, 9, 0, 9, 9],
 [0, 9, 0, 0, 0],
 [4, 4, 0, 0, 0],
 [4, 0, 4, 4, 4],
 [0, 4, 0, 0, 4]]
    )
    expected = ColoredGrid(values=
[[6, 6, 6, 6, 0], [0, 6, 6, 0, 0], [0, 0, 0, 0, 6]]
)
    actual = solve_31d5ba1a(input_grid)
    assert actual == expected


def test_31d5ba1a_example_4():
    input_grid = ColoredGrid(values=
[[0, 9, 9, 0, 0],
 [9, 0, 0, 0, 9],
 [9, 0, 0, 0, 0],
 [0, 0, 4, 0, 4],
 [4, 4, 0, 4, 0],
 [4, 0, 4, 4, 0]]
    )
    expected = ColoredGrid(values=
[[0, 6, 0, 0, 6], [0, 6, 0, 6, 6], [0, 0, 6, 6, 0]]
)
    actual = solve_31d5ba1a(input_grid)
    assert actual == expected



