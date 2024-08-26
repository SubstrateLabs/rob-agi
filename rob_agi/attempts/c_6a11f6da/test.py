import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_6a11f6da.main import solve_6a11f6da

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_6a11f6da_example_0():
    input_grid = ColoredGrid(values=
[[1, 0, 0, 0, 0],
 [1, 1, 1, 1, 1],
 [0, 1, 0, 1, 0],
 [1, 1, 1, 0, 1],
 [0, 0, 0, 1, 0],
 [8, 0, 8, 0, 0],
 [8, 0, 0, 8, 0],
 [8, 0, 0, 0, 8],
 [8, 8, 0, 0, 0],
 [8, 8, 0, 0, 0],
 [0, 6, 0, 0, 6],
 [6, 0, 0, 6, 6],
 [0, 6, 6, 6, 0],
 [6, 6, 0, 6, 6],
 [0, 0, 6, 0, 6]]
    )
    expected = ColoredGrid(values=
[[1, 6, 8, 0, 6],
 [6, 1, 1, 6, 6],
 [8, 6, 6, 6, 8],
 [6, 6, 1, 6, 6],
 [8, 8, 6, 1, 6]]
)
    actual = solve_6a11f6da(input_grid)
    assert actual == expected


def test_6a11f6da_example_1():
    input_grid = ColoredGrid(values=
[[1, 0, 1, 0, 1],
 [0, 1, 0, 0, 1],
 [0, 1, 0, 0, 0],
 [1, 0, 0, 1, 1],
 [1, 0, 0, 1, 1],
 [0, 0, 0, 0, 0],
 [0, 8, 8, 8, 0],
 [0, 8, 0, 0, 0],
 [8, 0, 0, 0, 8],
 [8, 0, 8, 8, 0],
 [0, 0, 6, 0, 6],
 [6, 0, 6, 0, 0],
 [6, 0, 0, 0, 6],
 [6, 0, 0, 0, 6],
 [0, 6, 6, 6, 6]]
    )
    expected = ColoredGrid(values=
[[1, 0, 6, 0, 6],
 [6, 1, 6, 8, 1],
 [6, 1, 0, 0, 6],
 [6, 0, 0, 1, 6],
 [1, 6, 6, 6, 6]]
)
    actual = solve_6a11f6da(input_grid)
    assert actual == expected


def test_6a11f6da_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 1, 1, 0],
 [1, 1, 1, 0, 0],
 [0, 1, 1, 1, 0],
 [0, 1, 0, 0, 1],
 [1, 0, 0, 1, 1],
 [8, 0, 8, 8, 0],
 [8, 0, 8, 8, 8],
 [8, 8, 8, 0, 8],
 [0, 8, 0, 8, 8],
 [8, 0, 8, 8, 8],
 [6, 0, 6, 0, 6],
 [0, 0, 0, 0, 6],
 [6, 6, 6, 6, 6],
 [0, 0, 6, 0, 0],
 [0, 6, 0, 6, 0]]
    )
    expected = ColoredGrid(values=
[[6, 0, 6, 1, 6],
 [1, 1, 1, 8, 6],
 [6, 6, 6, 6, 6],
 [0, 1, 6, 8, 1],
 [1, 6, 8, 6, 1]]
)
    actual = solve_6a11f6da(input_grid)
    assert actual == expected


def test_6a11f6da_example_3():
    input_grid = ColoredGrid(values=
[[0, 1, 1, 1, 1],
 [0, 1, 1, 0, 0],
 [0, 1, 1, 1, 0],
 [0, 0, 1, 1, 1],
 [0, 1, 1, 1, 0],
 [0, 8, 8, 0, 0],
 [8, 0, 0, 8, 0],
 [0, 8, 0, 0, 8],
 [0, 0, 8, 0, 0],
 [8, 0, 8, 0, 8],
 [0, 6, 0, 6, 6],
 [0, 0, 6, 6, 6],
 [0, 6, 0, 0, 0],
 [0, 6, 6, 0, 6],
 [0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 6, 1, 6, 6],
 [8, 1, 6, 6, 6],
 [0, 6, 1, 1, 8],
 [0, 6, 6, 1, 6],
 [8, 1, 1, 1, 8]]
)
    actual = solve_6a11f6da(input_grid)
    assert actual == expected


def test_6a11f6da_example_4():
    input_grid = ColoredGrid(values=
[[1, 1, 1, 0, 0],
 [0, 0, 1, 1, 0],
 [1, 1, 0, 0, 1],
 [0, 1, 1, 1, 1],
 [0, 0, 0, 0, 1],
 [0, 8, 0, 0, 8],
 [8, 8, 8, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 8, 0],
 [0, 0, 8, 8, 8],
 [6, 6, 0, 0, 0],
 [0, 6, 6, 6, 0],
 [0, 0, 6, 0, 6],
 [0, 0, 6, 6, 6],
 [6, 6, 6, 6, 6]]
    )
    expected = ColoredGrid(values=
[[6, 6, 1, 0, 8],
 [8, 6, 6, 6, 0],
 [1, 1, 6, 0, 6],
 [0, 1, 6, 6, 6],
 [6, 6, 6, 6, 6]]
)
    actual = solve_6a11f6da(input_grid)
    assert actual == expected



