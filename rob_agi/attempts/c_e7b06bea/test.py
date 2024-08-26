import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_e7b06bea.main import solve_e7b06bea

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_e7b06bea_example_0():
    input_grid = ColoredGrid(values=
[[5, 0, 0, 3, 1],
 [0, 0, 0, 3, 1],
 [0, 0, 0, 3, 1],
 [0, 0, 0, 3, 1],
 [0, 0, 0, 3, 1]]
    )
    expected = ColoredGrid(values=
[[5, 0, 3, 0, 0],
 [0, 0, 1, 0, 0],
 [0, 0, 3, 0, 0],
 [0, 0, 1, 0, 0],
 [0, 0, 3, 0, 0]]
)
    actual = solve_e7b06bea(input_grid)
    assert actual == expected


def test_e7b06bea_example_1():
    input_grid = ColoredGrid(values=
[[5, 0, 0, 0, 0, 9, 8],
 [5, 0, 0, 0, 0, 9, 8],
 [5, 0, 0, 0, 0, 9, 8],
 [0, 0, 0, 0, 0, 9, 8],
 [0, 0, 0, 0, 0, 9, 8],
 [0, 0, 0, 0, 0, 9, 8],
 [0, 0, 0, 0, 0, 9, 8]]
    )
    expected = ColoredGrid(values=
[[5, 0, 0, 0, 9, 0, 0],
 [5, 0, 0, 0, 9, 0, 0],
 [5, 0, 0, 0, 9, 0, 0],
 [0, 0, 0, 0, 8, 0, 0],
 [0, 0, 0, 0, 8, 0, 0],
 [0, 0, 0, 0, 8, 0, 0],
 [0, 0, 0, 0, 9, 0, 0]]
)
    actual = solve_e7b06bea(input_grid)
    assert actual == expected


def test_e7b06bea_example_2():
    input_grid = ColoredGrid(values=
[[5, 0, 0, 0, 9, 6, 7],
 [5, 0, 0, 0, 9, 6, 7],
 [0, 0, 0, 0, 9, 6, 7],
 [0, 0, 0, 0, 9, 6, 7],
 [0, 0, 0, 0, 9, 6, 7],
 [0, 0, 0, 0, 9, 6, 7],
 [0, 0, 0, 0, 9, 6, 7],
 [0, 0, 0, 0, 9, 6, 7],
 [0, 0, 0, 0, 9, 6, 7]]
    )
    expected = ColoredGrid(values=
[[5, 0, 0, 9, 0, 0, 0],
 [5, 0, 0, 9, 0, 0, 0],
 [0, 0, 0, 6, 0, 0, 0],
 [0, 0, 0, 6, 0, 0, 0],
 [0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 9, 0, 0, 0],
 [0, 0, 0, 9, 0, 0, 0],
 [0, 0, 0, 6, 0, 0, 0]]
)
    actual = solve_e7b06bea(input_grid)
    assert actual == expected


def test_e7b06bea_example_3():
    input_grid = ColoredGrid(values=
[[5, 0, 0, 0, 0, 0, 2, 3],
 [5, 0, 0, 0, 0, 0, 2, 3],
 [5, 0, 0, 0, 0, 0, 2, 3],
 [5, 0, 0, 0, 0, 0, 2, 3],
 [0, 0, 0, 0, 0, 0, 2, 3],
 [0, 0, 0, 0, 0, 0, 2, 3],
 [0, 0, 0, 0, 0, 0, 2, 3],
 [0, 0, 0, 0, 0, 0, 2, 3],
 [0, 0, 0, 0, 0, 0, 2, 3],
 [0, 0, 0, 0, 0, 0, 2, 3],
 [0, 0, 0, 0, 0, 0, 2, 3],
 [0, 0, 0, 0, 0, 0, 2, 3]]
    )
    expected = ColoredGrid(values=
[[5, 0, 0, 0, 0, 2, 0, 0],
 [5, 0, 0, 0, 0, 2, 0, 0],
 [5, 0, 0, 0, 0, 2, 0, 0],
 [5, 0, 0, 0, 0, 2, 0, 0],
 [0, 0, 0, 0, 0, 3, 0, 0],
 [0, 0, 0, 0, 0, 3, 0, 0],
 [0, 0, 0, 0, 0, 3, 0, 0],
 [0, 0, 0, 0, 0, 3, 0, 0],
 [0, 0, 0, 0, 0, 2, 0, 0],
 [0, 0, 0, 0, 0, 2, 0, 0],
 [0, 0, 0, 0, 0, 2, 0, 0],
 [0, 0, 0, 0, 0, 2, 0, 0]]
)
    actual = solve_e7b06bea(input_grid)
    assert actual == expected


def test_e7b06bea_example_4():
    input_grid = ColoredGrid(values=
[[5, 0, 0, 2, 8, 4],
 [0, 0, 0, 2, 8, 4],
 [0, 0, 0, 2, 8, 4],
 [0, 0, 0, 2, 8, 4],
 [0, 0, 0, 2, 8, 4],
 [0, 0, 0, 2, 8, 4],
 [0, 0, 0, 2, 8, 4],
 [0, 0, 0, 2, 8, 4],
 [0, 0, 0, 2, 8, 4],
 [0, 0, 0, 2, 8, 4],
 [0, 0, 0, 2, 8, 4],
 [0, 0, 0, 2, 8, 4],
 [0, 0, 0, 2, 8, 4],
 [0, 0, 0, 2, 8, 4]]
    )
    expected = ColoredGrid(values=
[[5, 0, 2, 0, 0, 0],
 [0, 0, 8, 0, 0, 0],
 [0, 0, 4, 0, 0, 0],
 [0, 0, 2, 0, 0, 0],
 [0, 0, 8, 0, 0, 0],
 [0, 0, 4, 0, 0, 0],
 [0, 0, 2, 0, 0, 0],
 [0, 0, 8, 0, 0, 0],
 [0, 0, 4, 0, 0, 0],
 [0, 0, 2, 0, 0, 0],
 [0, 0, 8, 0, 0, 0],
 [0, 0, 4, 0, 0, 0],
 [0, 0, 2, 0, 0, 0],
 [0, 0, 8, 0, 0, 0]]
)
    actual = solve_e7b06bea(input_grid)
    assert actual == expected



