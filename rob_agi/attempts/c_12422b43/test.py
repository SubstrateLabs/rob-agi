import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_12422b43.main import solve_12422b43

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_12422b43_example_0():
    input_grid = ColoredGrid(values=
[[5, 0, 6, 0, 0],
 [5, 4, 4, 4, 0],
 [0, 0, 6, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[5, 0, 6, 0, 0],
 [5, 4, 4, 4, 0],
 [0, 0, 6, 0, 0],
 [0, 0, 6, 0, 0],
 [0, 4, 4, 4, 0]]
)
    actual = solve_12422b43(input_grid)
    assert actual == expected


def test_12422b43_example_1():
    input_grid = ColoredGrid(values=
[[5, 0, 8, 8, 0, 0, 0],
 [5, 0, 0, 7, 0, 0, 0],
 [5, 0, 0, 4, 4, 0, 0],
 [0, 0, 3, 3, 0, 0, 0],
 [0, 0, 1, 1, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[5, 0, 8, 8, 0, 0, 0],
 [5, 0, 0, 7, 0, 0, 0],
 [5, 0, 0, 4, 4, 0, 0],
 [0, 0, 3, 3, 0, 0, 0],
 [0, 0, 1, 1, 0, 0, 0],
 [0, 0, 8, 8, 0, 0, 0],
 [0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 4, 4, 0, 0]]
)
    actual = solve_12422b43(input_grid)
    assert actual == expected


def test_12422b43_example_2():
    input_grid = ColoredGrid(values=
[[5, 0, 0, 4, 4, 0, 0],
 [5, 0, 8, 8, 8, 0, 0],
 [0, 0, 0, 2, 0, 0, 0],
 [0, 0, 0, 3, 3, 0, 0],
 [0, 0, 4, 4, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[5, 0, 0, 4, 4, 0, 0],
 [5, 0, 8, 8, 8, 0, 0],
 [0, 0, 0, 2, 0, 0, 0],
 [0, 0, 0, 3, 3, 0, 0],
 [0, 0, 4, 4, 0, 0, 0],
 [0, 0, 0, 4, 4, 0, 0],
 [0, 0, 8, 8, 8, 0, 0],
 [0, 0, 0, 4, 4, 0, 0],
 [0, 0, 8, 8, 8, 0, 0]]
)
    actual = solve_12422b43(input_grid)
    assert actual == expected


def test_12422b43_example_3():
    input_grid = ColoredGrid(values=
[[5, 0, 0, 3, 3, 0],
 [5, 0, 0, 3, 2, 0],
 [5, 0, 0, 2, 3, 0],
 [5, 0, 0, 8, 8, 0],
 [0, 0, 0, 8, 8, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[5, 0, 0, 3, 3, 0],
 [5, 0, 0, 3, 2, 0],
 [5, 0, 0, 2, 3, 0],
 [5, 0, 0, 8, 8, 0],
 [0, 0, 0, 8, 8, 0],
 [0, 0, 0, 3, 3, 0],
 [0, 0, 0, 3, 2, 0],
 [0, 0, 0, 2, 3, 0],
 [0, 0, 0, 8, 8, 0],
 [0, 0, 0, 3, 3, 0],
 [0, 0, 0, 3, 2, 0],
 [0, 0, 0, 2, 3, 0],
 [0, 0, 0, 8, 8, 0]]
)
    actual = solve_12422b43(input_grid)
    assert actual == expected


def test_12422b43_example_4():
    input_grid = ColoredGrid(values=
[[5, 0, 6, 8, 0, 0],
 [0, 0, 8, 3, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[5, 0, 6, 8, 0, 0],
 [0, 0, 8, 3, 0, 0],
 [0, 0, 6, 8, 0, 0],
 [0, 0, 6, 8, 0, 0],
 [0, 0, 6, 8, 0, 0],
 [0, 0, 6, 8, 0, 0],
 [0, 0, 6, 8, 0, 0]]
)
    actual = solve_12422b43(input_grid)
    assert actual == expected



