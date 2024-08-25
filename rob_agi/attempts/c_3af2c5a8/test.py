import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_3af2c5a8.main import solve_3af2c5a8


def test_3af2c5a8_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 8, 0], [0, 8, 0, 8], [0, 0, 8, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 8, 0, 0, 8, 0, 0],
 [0, 8, 0, 8, 8, 0, 8, 0],
 [0, 0, 8, 0, 0, 8, 0, 0],
 [0, 0, 8, 0, 0, 8, 0, 0],
 [0, 8, 0, 8, 8, 0, 8, 0],
 [0, 0, 8, 0, 0, 8, 0, 0]]
)
    actual = solve_3af2c5a8(input_grid)
    assert actual == expected


def test_3af2c5a8_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 3, 3], [0, 3, 0, 3], [3, 3, 3, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 3, 3, 3, 3, 0, 0],
 [0, 3, 0, 3, 3, 0, 3, 0],
 [3, 3, 3, 0, 0, 3, 3, 3],
 [3, 3, 3, 0, 0, 3, 3, 3],
 [0, 3, 0, 3, 3, 0, 3, 0],
 [0, 0, 3, 3, 3, 3, 0, 0]]
)
    actual = solve_3af2c5a8(input_grid)
    assert actual == expected


def test_3af2c5a8_example_2():
    input_grid = ColoredGrid(values=
[[3, 3, 3, 3], [3, 0, 0, 0], [3, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[3, 3, 3, 3, 3, 3, 3, 3],
 [3, 0, 0, 0, 0, 0, 0, 3],
 [3, 0, 0, 0, 0, 0, 0, 3],
 [3, 0, 0, 0, 0, 0, 0, 3],
 [3, 0, 0, 0, 0, 0, 0, 3],
 [3, 3, 3, 3, 3, 3, 3, 3]]
)
    actual = solve_3af2c5a8(input_grid)
    assert actual == expected



def test_3af2c5a8_test_case_0():
    input_grid = ColoredGrid(values=
[[4, 0, 0, 0], [0, 0, 0, 4], [4, 4, 0, 0]]
    )
    expected = ColoredGrid(values=
[[4, 0, 0, 0, 0, 0, 0, 4],
 [0, 0, 0, 4, 4, 0, 0, 0],
 [4, 4, 0, 0, 0, 0, 4, 4],
 [4, 4, 0, 0, 0, 0, 4, 4],
 [0, 0, 0, 4, 4, 0, 0, 0],
 [4, 0, 0, 0, 0, 0, 0, 4]]
    )
    actual = solve_3af2c5a8(input_grid)
    assert actual == expected

