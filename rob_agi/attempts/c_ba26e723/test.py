import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_ba26e723.main import solve_ba26e723


def test_ba26e723_example_0():
    input_grid = ColoredGrid(values=
[[4, 0, 4, 0, 4, 0, 4, 0, 4, 0],
 [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
 [0, 4, 0, 4, 0, 4, 0, 4, 0, 4]]
    )
    expected = ColoredGrid(values=
[[6, 0, 4, 0, 4, 0, 6, 0, 4, 0],
 [6, 4, 4, 6, 4, 4, 6, 4, 4, 6],
 [0, 4, 0, 6, 0, 4, 0, 4, 0, 6]]
)
    actual = solve_ba26e723(input_grid)
    assert actual == expected


def test_ba26e723_example_1():
    input_grid = ColoredGrid(values=
[[0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0],
 [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
 [4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4]]
    )
    expected = ColoredGrid(values=
[[0, 4, 0, 6, 0, 4, 0, 4, 0, 6, 0],
 [6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4],
 [6, 0, 4, 0, 4, 0, 6, 0, 4, 0, 4]]
)
    actual = solve_ba26e723(input_grid)
    assert actual == expected


def test_ba26e723_example_2():
    input_grid = ColoredGrid(values=
[[4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4],
 [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
 [0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0]]
    )
    expected = ColoredGrid(values=
[[6, 0, 4, 0, 4, 0, 6, 0, 4, 0, 4],
 [6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4],
 [0, 4, 0, 6, 0, 4, 0, 4, 0, 6, 0]]
)
    actual = solve_ba26e723(input_grid)
    assert actual == expected


def test_ba26e723_example_3():
    input_grid = ColoredGrid(values=
[[4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4],
 [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
 [0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0]]
    )
    expected = ColoredGrid(values=
[[6, 0, 4, 0, 4, 0, 6, 0, 4, 0, 4, 0, 6],
 [6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4, 6],
 [0, 4, 0, 6, 0, 4, 0, 4, 0, 6, 0, 4, 0]]
)
    actual = solve_ba26e723(input_grid)
    assert actual == expected


def test_ba26e723_example_4():
    input_grid = ColoredGrid(values=
[[0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4],
 [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
 [4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0]]
    )
    expected = ColoredGrid(values=
[[0, 4, 0, 6, 0, 4, 0, 4, 0, 6, 0, 4, 0, 4],
 [6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4],
 [6, 0, 4, 0, 4, 0, 6, 0, 4, 0, 4, 0, 6, 0]]
)
    actual = solve_ba26e723(input_grid)
    assert actual == expected



def test_ba26e723_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0],
 [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
 [4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4]]
    )
    expected = ColoredGrid(values=
[[0, 4, 0, 6, 0, 4, 0, 4, 0, 6, 0, 4, 0, 4, 0, 6, 0],
 [6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4],
 [6, 0, 4, 0, 4, 0, 6, 0, 4, 0, 4, 0, 6, 0, 4, 0, 4]]
    )
    actual = solve_ba26e723(input_grid)
    assert actual == expected

