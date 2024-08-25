import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_b1948b0a.main import solve_b1948b0a


def test_b1948b0a_example_0():
    input_grid = ColoredGrid(values=
[[6, 6, 7, 6], [6, 6, 7, 7], [7, 7, 6, 7]]
    )
    expected = ColoredGrid(values=
[[2, 2, 7, 2], [2, 2, 7, 7], [7, 7, 2, 7]]
)
    actual = solve_b1948b0a(input_grid)
    assert actual == expected


def test_b1948b0a_example_1():
    input_grid = ColoredGrid(values=
[[7, 7, 7, 6],
 [6, 6, 7, 6],
 [7, 7, 6, 7],
 [7, 6, 7, 7],
 [7, 6, 7, 6],
 [6, 6, 6, 7]]
    )
    expected = ColoredGrid(values=
[[7, 7, 7, 2],
 [2, 2, 7, 2],
 [7, 7, 2, 7],
 [7, 2, 7, 7],
 [7, 2, 7, 2],
 [2, 2, 2, 7]]
)
    actual = solve_b1948b0a(input_grid)
    assert actual == expected


def test_b1948b0a_example_2():
    input_grid = ColoredGrid(values=
[[7, 7, 6, 6, 6, 6], [6, 7, 6, 7, 7, 7], [7, 6, 7, 7, 6, 7]]
    )
    expected = ColoredGrid(values=
[[7, 7, 2, 2, 2, 2], [2, 7, 2, 7, 7, 7], [7, 2, 7, 7, 2, 7]]
)
    actual = solve_b1948b0a(input_grid)
    assert actual == expected



def test_b1948b0a_test_case_0():
    input_grid = ColoredGrid(values=
[[6, 7, 7, 6], [6, 7, 6, 7], [7, 7, 7, 6], [7, 6, 7, 6]]
    )
    expected = ColoredGrid(values=
[[2, 7, 7, 2], [2, 7, 2, 7], [7, 7, 7, 2], [7, 2, 7, 2]]
    )
    actual = solve_b1948b0a(input_grid)
    assert actual == expected

