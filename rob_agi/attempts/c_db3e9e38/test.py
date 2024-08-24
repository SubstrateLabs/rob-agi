import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_db3e9e38.main import solve_db3e9e38


def test_db3e9e38_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 7, 8, 7, 8, 7, 8],
 [0, 7, 8, 7, 8, 7, 0],
 [0, 0, 8, 7, 8, 0, 0],
 [0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_db3e9e38(input_grid)
    assert actual == expected


def test_db3e9e38_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 7, 0, 0, 0, 0, 0],
 [0, 0, 7, 0, 0, 0, 0, 0],
 [0, 0, 7, 0, 0, 0, 0, 0],
 [0, 0, 7, 0, 0, 0, 0, 0],
 [0, 0, 7, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[7, 8, 7, 8, 7, 8, 7, 0],
 [7, 8, 7, 8, 7, 8, 0, 0],
 [7, 8, 7, 8, 7, 0, 0, 0],
 [0, 8, 7, 8, 0, 0, 0, 0],
 [0, 0, 7, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_db3e9e38(input_grid)
    assert actual == expected



def test_db3e9e38_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 7, 8, 7, 8, 7, 8, 7, 8],
 [8, 7, 8, 7, 8, 7, 8, 7, 8],
 [0, 7, 8, 7, 8, 7, 8, 7, 8],
 [0, 0, 8, 7, 8, 7, 8, 7, 8],
 [0, 0, 0, 7, 8, 7, 8, 7, 0],
 [0, 0, 0, 0, 8, 7, 8, 0, 0],
 [0, 0, 0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    actual = solve_db3e9e38(input_grid)
    assert actual == expected

