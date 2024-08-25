import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_fc754716.main import solve_fc754716


def test_fc754716_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0], [0, 2, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 2, 2], [2, 0, 2], [2, 2, 2]]
)
    actual = solve_fc754716(input_grid)
    assert actual == expected


def test_fc754716_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0], [0, 0, 0], [0, 3, 0], [0, 0, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[3, 3, 3], [3, 0, 3], [3, 0, 3], [3, 0, 3], [3, 3, 3]]
)
    actual = solve_fc754716(input_grid)
    assert actual == expected


def test_fc754716_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 1, 1, 1, 1, 1, 1],
 [1, 0, 0, 0, 0, 0, 1],
 [1, 0, 0, 0, 0, 0, 1],
 [1, 0, 0, 0, 0, 0, 1],
 [1, 1, 1, 1, 1, 1, 1]]
)
    actual = solve_fc754716(input_grid)
    assert actual == expected


def test_fc754716_example_3():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 6, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[6, 6, 6, 6, 6],
 [6, 0, 0, 0, 6],
 [6, 0, 0, 0, 6],
 [6, 0, 0, 0, 6],
 [6, 6, 6, 6, 6]]
)
    actual = solve_fc754716(input_grid)
    assert actual == expected



def test_fc754716_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 8, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 8, 8, 8, 8, 8, 8],
 [8, 0, 0, 0, 0, 0, 8],
 [8, 0, 0, 0, 0, 0, 8],
 [8, 0, 0, 0, 0, 0, 8],
 [8, 0, 0, 0, 0, 0, 8],
 [8, 0, 0, 0, 0, 0, 8],
 [8, 0, 0, 0, 0, 0, 8],
 [8, 0, 0, 0, 0, 0, 8],
 [8, 8, 8, 8, 8, 8, 8]]
    )
    actual = solve_fc754716(input_grid)
    assert actual == expected

