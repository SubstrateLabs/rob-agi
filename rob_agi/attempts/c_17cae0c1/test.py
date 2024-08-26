import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_17cae0c1.main import solve_17cae0c1

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_17cae0c1_example_0():
    input_grid = ColoredGrid(values=
[[5, 5, 5, 0, 0, 0, 0, 0, 5],
 [5, 0, 5, 0, 5, 0, 0, 5, 0],
 [5, 5, 5, 0, 0, 0, 5, 0, 0]]
    )
    expected = ColoredGrid(values=
[[3, 3, 3, 4, 4, 4, 9, 9, 9],
 [3, 3, 3, 4, 4, 4, 9, 9, 9],
 [3, 3, 3, 4, 4, 4, 9, 9, 9]]
)
    actual = solve_17cae0c1(input_grid)
    assert actual == expected


def test_17cae0c1_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 5, 0, 0, 0, 0, 0, 0],
 [0, 5, 0, 0, 0, 0, 0, 5, 0],
 [5, 0, 0, 5, 5, 5, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[9, 9, 9, 1, 1, 1, 4, 4, 4],
 [9, 9, 9, 1, 1, 1, 4, 4, 4],
 [9, 9, 9, 1, 1, 1, 4, 4, 4]]
)
    actual = solve_17cae0c1(input_grid)
    assert actual == expected


def test_17cae0c1_example_2():
    input_grid = ColoredGrid(values=
[[5, 5, 5, 5, 5, 5, 0, 0, 0],
 [0, 0, 0, 5, 0, 5, 0, 0, 0],
 [0, 0, 0, 5, 5, 5, 5, 5, 5]]
    )
    expected = ColoredGrid(values=
[[6, 6, 6, 3, 3, 3, 1, 1, 1],
 [6, 6, 6, 3, 3, 3, 1, 1, 1],
 [6, 6, 6, 3, 3, 3, 1, 1, 1]]
)
    actual = solve_17cae0c1(input_grid)
    assert actual == expected


def test_17cae0c1_example_3():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 5, 5, 5, 5, 5, 5],
 [0, 5, 0, 0, 0, 0, 5, 0, 5],
 [0, 0, 0, 0, 0, 0, 5, 5, 5]]
    )
    expected = ColoredGrid(values=
[[4, 4, 4, 6, 6, 6, 3, 3, 3],
 [4, 4, 4, 6, 6, 6, 3, 3, 3],
 [4, 4, 4, 6, 6, 6, 3, 3, 3]]
)
    actual = solve_17cae0c1(input_grid)
    assert actual == expected



