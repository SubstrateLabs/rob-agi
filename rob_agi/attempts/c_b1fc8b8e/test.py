import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_b1fc8b8e.main import solve_b1fc8b8e


def test_b1fc8b8e_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 8, 0, 0],
 [0, 0, 8, 8, 8, 0],
 [0, 8, 0, 8, 8, 0],
 [8, 8, 8, 0, 0, 0],
 [0, 8, 8, 0, 0, 0],
 [0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 8, 0, 0, 8],
 [8, 8, 0, 8, 8],
 [0, 0, 0, 0, 0],
 [0, 8, 0, 0, 8],
 [8, 8, 0, 8, 8]]
)
    actual = solve_b1fc8b8e(input_grid)
    assert actual == expected


def test_b1fc8b8e_example_1():
    input_grid = ColoredGrid(values=
[[8, 8, 8, 8, 0, 0],
 [8, 8, 8, 8, 8, 8],
 [0, 8, 8, 0, 8, 8],
 [0, 8, 8, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 8, 0, 8, 8],
 [8, 8, 0, 8, 8],
 [0, 0, 0, 0, 0],
 [8, 8, 0, 8, 8],
 [8, 8, 0, 8, 8]]
)
    actual = solve_b1fc8b8e(input_grid)
    assert actual == expected


def test_b1fc8b8e_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 8, 0, 0],
 [0, 8, 8, 8, 8, 0],
 [8, 8, 8, 8, 8, 0],
 [0, 8, 8, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 8, 0, 0, 8],
 [8, 8, 0, 8, 8],
 [0, 0, 0, 0, 0],
 [0, 8, 0, 0, 8],
 [8, 8, 0, 8, 8]]
)
    actual = solve_b1fc8b8e(input_grid)
    assert actual == expected


def test_b1fc8b8e_example_3():
    input_grid = ColoredGrid(values=
[[0, 0, 8, 8, 0, 0],
 [8, 8, 8, 8, 0, 0],
 [8, 8, 8, 8, 8, 8],
 [0, 0, 8, 8, 8, 8],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 8, 0, 8, 8],
 [8, 8, 0, 8, 8],
 [0, 0, 0, 0, 0],
 [8, 8, 0, 8, 8],
 [8, 8, 0, 8, 8]]
)
    actual = solve_b1fc8b8e(input_grid)
    assert actual == expected


def test_b1fc8b8e_example_4():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 8, 0, 0],
 [0, 8, 8, 8, 0, 0],
 [8, 8, 8, 0, 8, 0],
 [0, 8, 8, 8, 8, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 8, 0, 0, 8],
 [8, 8, 0, 8, 8],
 [0, 0, 0, 0, 0],
 [0, 8, 0, 0, 8],
 [8, 8, 0, 8, 8]]
)
    actual = solve_b1fc8b8e(input_grid)
    assert actual == expected



def test_b1fc8b8e_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 8, 8, 0, 0],
 [8, 8, 8, 8, 0, 0],
 [8, 8, 0, 8, 8, 0],
 [0, 8, 8, 8, 8, 0],
 [0, 8, 8, 0, 0, 0],
 [0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 8, 0, 8, 8],
 [8, 8, 0, 8, 8],
 [0, 0, 0, 0, 0],
 [8, 8, 0, 8, 8],
 [8, 8, 0, 8, 8]]
    )
    actual = solve_b1fc8b8e(input_grid)
    assert actual == expected


def test_b1fc8b8e_test_case_1():
    input_grid = ColoredGrid(values=
[[0, 8, 0, 8, 0, 0],
 [8, 8, 8, 8, 8, 0],
 [0, 0, 0, 8, 8, 8],
 [0, 0, 0, 0, 8, 8],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 8, 0, 0, 8],
 [8, 8, 0, 8, 8],
 [0, 0, 0, 0, 0],
 [0, 8, 0, 0, 8],
 [8, 8, 0, 8, 8]]
    )
    actual = solve_b1fc8b8e(input_grid)
    assert actual == expected

