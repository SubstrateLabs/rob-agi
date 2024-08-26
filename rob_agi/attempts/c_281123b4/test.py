import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_281123b4.main import solve_281123b4

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_281123b4_example_0():
    input_grid = ColoredGrid(values=
[[8, 8, 8, 0, 3, 5, 5, 5, 0, 3, 9, 9, 9, 0, 3, 4, 4, 4, 4],
 [8, 0, 8, 0, 3, 5, 5, 5, 5, 3, 9, 9, 0, 9, 3, 0, 0, 0, 0],
 [0, 0, 0, 8, 3, 5, 5, 0, 0, 3, 0, 0, 0, 0, 3, 0, 4, 4, 0],
 [0, 8, 0, 0, 3, 0, 5, 5, 5, 3, 9, 0, 0, 0, 3, 4, 4, 4, 4]]
    )
    expected = ColoredGrid(values=
[[9, 9, 9, 4], [9, 9, 8, 9], [5, 4, 4, 8], [9, 4, 4, 4]]
)
    actual = solve_281123b4(input_grid)
    assert actual == expected


def test_281123b4_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 8, 8, 3, 5, 5, 0, 0, 3, 0, 9, 9, 9, 3, 4, 0, 4, 0],
 [8, 8, 8, 8, 3, 0, 5, 0, 5, 3, 0, 9, 0, 9, 3, 4, 0, 4, 0],
 [8, 8, 0, 8, 3, 5, 0, 5, 5, 3, 0, 0, 0, 9, 3, 0, 4, 0, 4],
 [0, 8, 8, 0, 3, 0, 0, 0, 5, 3, 9, 0, 0, 9, 3, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[4, 9, 9, 9], [4, 9, 4, 9], [8, 4, 5, 9], [9, 8, 8, 9]]
)
    actual = solve_281123b4(input_grid)
    assert actual == expected


def test_281123b4_example_2():
    input_grid = ColoredGrid(values=
[[8, 8, 0, 0, 3, 5, 5, 5, 0, 3, 9, 0, 9, 9, 3, 4, 4, 0, 4],
 [8, 8, 0, 8, 3, 5, 5, 5, 5, 3, 0, 9, 0, 0, 3, 0, 0, 4, 4],
 [8, 0, 0, 0, 3, 0, 5, 0, 5, 3, 9, 0, 0, 9, 3, 4, 0, 0, 4],
 [8, 0, 8, 8, 3, 5, 0, 5, 0, 3, 0, 0, 0, 0, 3, 0, 0, 4, 0]]
    )
    expected = ColoredGrid(values=
[[9, 4, 9, 9], [8, 9, 4, 4], [9, 5, 0, 9], [8, 0, 4, 8]]
)
    actual = solve_281123b4(input_grid)
    assert actual == expected


def test_281123b4_example_3():
    input_grid = ColoredGrid(values=
[[0, 0, 8, 8, 3, 5, 0, 0, 5, 3, 9, 0, 0, 9, 3, 4, 0, 0, 4],
 [0, 8, 8, 0, 3, 5, 5, 0, 5, 3, 9, 9, 0, 9, 3, 0, 0, 4, 4],
 [8, 8, 8, 0, 3, 0, 5, 5, 0, 3, 9, 9, 0, 0, 3, 4, 0, 0, 0],
 [8, 8, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 4, 4, 4, 0]]
    )
    expected = ColoredGrid(values=
[[9, 0, 8, 9], [9, 9, 4, 9], [9, 9, 8, 0], [4, 4, 4, 0]]
)
    actual = solve_281123b4(input_grid)
    assert actual == expected


def test_281123b4_example_4():
    input_grid = ColoredGrid(values=
[[0, 8, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 9, 3, 4, 0, 4, 0],
 [0, 8, 0, 0, 3, 5, 5, 0, 0, 3, 0, 9, 9, 0, 3, 4, 0, 0, 4],
 [8, 8, 8, 0, 3, 5, 0, 0, 5, 3, 9, 9, 9, 0, 3, 4, 0, 4, 0],
 [0, 0, 0, 0, 3, 5, 5, 5, 5, 3, 0, 0, 9, 0, 3, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[4, 8, 4, 9], [4, 9, 9, 4], [9, 9, 9, 5], [5, 5, 9, 5]]
)
    actual = solve_281123b4(input_grid)
    assert actual == expected


def test_281123b4_example_5():
    input_grid = ColoredGrid(values=
[[0, 8, 8, 0, 3, 5, 5, 5, 5, 3, 9, 9, 0, 9, 3, 4, 0, 0, 4],
 [8, 0, 8, 0, 3, 0, 5, 0, 5, 3, 0, 0, 0, 9, 3, 4, 0, 4, 4],
 [8, 8, 0, 8, 3, 0, 0, 0, 0, 3, 9, 9, 0, 9, 3, 0, 4, 0, 4],
 [8, 8, 0, 8, 3, 5, 5, 0, 0, 3, 9, 9, 0, 0, 3, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[9, 9, 8, 9], [4, 5, 4, 9], [9, 9, 0, 9], [9, 9, 0, 8]]
)
    actual = solve_281123b4(input_grid)
    assert actual == expected



