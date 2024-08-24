import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_d90796e8.main import solve_d90796e8


def test_d90796e8_example_0():
    input_grid = ColoredGrid(values=
[[3, 2, 0], [0, 0, 0], [0, 5, 0]]
    )
    expected = ColoredGrid(values=
[[8, 0, 0], [0, 0, 0], [0, 5, 0]]
)
    actual = solve_d90796e8(input_grid)
    assert actual == expected


def test_d90796e8_example_1():
    input_grid = ColoredGrid(values=
[[5, 0, 0, 0, 0, 0],
 [0, 0, 3, 2, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 3, 0, 0, 0, 2],
 [0, 2, 0, 0, 0, 0],
 [5, 0, 0, 3, 0, 0],
 [0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[5, 0, 0, 0, 0, 0],
 [0, 0, 8, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 8, 0, 0, 0, 2],
 [0, 0, 0, 0, 0, 0],
 [5, 0, 0, 3, 0, 0],
 [0, 0, 0, 0, 0, 0]]
)
    actual = solve_d90796e8(input_grid)
    assert actual == expected


def test_d90796e8_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 2, 0],
 [3, 0, 0, 0, 0, 0, 3],
 [5, 0, 2, 3, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 2, 0],
 [3, 2, 0, 0, 0, 3, 0],
 [0, 0, 0, 5, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 2, 0],
 [3, 0, 0, 0, 0, 0, 3],
 [5, 0, 0, 8, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [8, 0, 0, 0, 0, 8, 0],
 [0, 0, 0, 5, 0, 0, 0]]
)
    actual = solve_d90796e8(input_grid)
    assert actual == expected



def test_d90796e8_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 2, 0, 0, 0, 5],
 [0, 2, 0, 0, 0, 0, 3, 2, 0],
 [0, 3, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 5, 0, 0, 0, 2],
 [5, 0, 0, 2, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 3, 0],
 [5, 3, 0, 0, 0, 5, 0, 2, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 2, 0, 0, 0, 5],
 [0, 0, 0, 0, 0, 0, 8, 0, 0],
 [0, 8, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 5, 0, 0, 0, 2],
 [5, 0, 0, 2, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 8, 0],
 [5, 3, 0, 0, 0, 5, 0, 0, 0]]
    )
    actual = solve_d90796e8(input_grid)
    assert actual == expected

