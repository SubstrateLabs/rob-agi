import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_3d31c5b3.main import solve_3d31c5b3

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_3d31c5b3_example_0():
    input_grid = ColoredGrid(values=
[[0, 5, 5, 5, 5, 0],
 [5, 5, 0, 5, 5, 5],
 [5, 5, 0, 5, 0, 0],
 [0, 0, 4, 0, 0, 0],
 [4, 0, 4, 4, 4, 0],
 [4, 0, 0, 0, 0, 0],
 [2, 0, 2, 2, 0, 2],
 [2, 0, 0, 0, 0, 2],
 [0, 0, 0, 2, 0, 0],
 [0, 8, 0, 8, 0, 0],
 [0, 8, 0, 0, 0, 0],
 [0, 8, 0, 8, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 5, 5, 5, 5, 2], [5, 5, 4, 5, 5, 5], [5, 5, 0, 5, 0, 0]]
)
    actual = solve_3d31c5b3(input_grid)
    assert actual == expected


def test_3d31c5b3_example_1():
    input_grid = ColoredGrid(values=
[[5, 5, 0, 5, 5, 5],
 [0, 5, 0, 5, 0, 5],
 [0, 0, 0, 5, 5, 0],
 [0, 4, 4, 0, 4, 0],
 [0, 0, 0, 0, 0, 4],
 [0, 4, 0, 4, 0, 4],
 [2, 2, 2, 0, 0, 0],
 [0, 2, 2, 0, 2, 0],
 [2, 2, 2, 0, 2, 0],
 [8, 0, 8, 8, 8, 8],
 [0, 0, 8, 8, 8, 8],
 [0, 0, 0, 8, 0, 0]]
    )
    expected = ColoredGrid(values=
[[5, 5, 4, 5, 5, 5], [0, 5, 8, 5, 8, 5], [2, 4, 2, 5, 5, 4]]
)
    actual = solve_3d31c5b3(input_grid)
    assert actual == expected


def test_3d31c5b3_example_2():
    input_grid = ColoredGrid(values=
[[5, 0, 5, 0, 0, 0],
 [0, 0, 5, 0, 0, 5],
 [5, 0, 5, 0, 5, 0],
 [0, 0, 0, 4, 0, 4],
 [0, 0, 0, 4, 0, 0],
 [4, 0, 0, 4, 0, 4],
 [0, 0, 2, 0, 0, 2],
 [2, 2, 0, 2, 2, 0],
 [2, 2, 0, 0, 0, 2],
 [8, 8, 0, 8, 8, 8],
 [8, 8, 8, 8, 8, 0],
 [8, 8, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[5, 8, 5, 4, 8, 4], [8, 8, 5, 4, 8, 5], [5, 8, 5, 4, 5, 4]]
)
    actual = solve_3d31c5b3(input_grid)
    assert actual == expected


def test_3d31c5b3_example_3():
    input_grid = ColoredGrid(values=
[[5, 5, 5, 5, 0, 0],
 [0, 5, 5, 0, 5, 5],
 [0, 5, 5, 5, 5, 5],
 [4, 4, 4, 0, 4, 4],
 [0, 0, 0, 4, 4, 0],
 [4, 4, 4, 0, 4, 0],
 [2, 0, 2, 2, 0, 0],
 [2, 2, 0, 2, 0, 0],
 [2, 2, 2, 0, 2, 0],
 [0, 0, 8, 0, 8, 8],
 [8, 8, 8, 0, 0, 0],
 [0, 8, 0, 0, 8, 0]]
    )
    expected = ColoredGrid(values=
[[5, 5, 5, 5, 4, 4], [8, 5, 5, 4, 5, 5], [4, 5, 5, 5, 5, 5]]
)
    actual = solve_3d31c5b3(input_grid)
    assert actual == expected


def test_3d31c5b3_example_4():
    input_grid = ColoredGrid(values=
[[5, 0, 0, 0, 0, 0],
 [0, 5, 0, 0, 0, 5],
 [0, 0, 5, 5, 5, 0],
 [4, 4, 0, 4, 4, 4],
 [0, 0, 0, 4, 4, 0],
 [4, 0, 4, 4, 0, 0],
 [2, 0, 2, 2, 0, 2],
 [2, 2, 0, 2, 2, 0],
 [0, 0, 0, 0, 0, 2],
 [8, 8, 8, 8, 0, 8],
 [0, 0, 0, 8, 8, 0],
 [0, 0, 0, 8, 8, 8]]
    )
    expected = ColoredGrid(values=
[[5, 4, 8, 4, 4, 4], [2, 5, 0, 4, 4, 5], [4, 0, 5, 5, 5, 8]]
)
    actual = solve_3d31c5b3(input_grid)
    assert actual == expected


def test_3d31c5b3_example_5():
    input_grid = ColoredGrid(values=
[[0, 5, 0, 5, 5, 0],
 [0, 5, 0, 5, 5, 5],
 [5, 5, 0, 5, 5, 5],
 [4, 0, 0, 0, 4, 4],
 [0, 0, 0, 4, 4, 0],
 [4, 0, 4, 0, 0, 4],
 [0, 2, 2, 2, 2, 0],
 [2, 2, 2, 0, 2, 0],
 [0, 2, 0, 2, 0, 0],
 [8, 0, 0, 8, 0, 8],
 [8, 0, 0, 0, 8, 0],
 [8, 0, 0, 8, 0, 0]]
    )
    expected = ColoredGrid(values=
[[4, 5, 2, 5, 5, 4], [8, 5, 2, 5, 5, 5], [5, 5, 4, 5, 5, 5]]
)
    actual = solve_3d31c5b3(input_grid)
    assert actual == expected



