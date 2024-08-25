import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_6f8cd79b.main import solve_6f8cd79b


def test_6f8cd79b_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 8, 8], [8, 0, 8], [8, 8, 8]]
)
    actual = solve_6f8cd79b(input_grid)
    assert actual == expected


def test_6f8cd79b_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 8, 8], [8, 0, 8], [8, 0, 8], [8, 8, 8]]
)
    actual = solve_6f8cd79b(input_grid)
    assert actual == expected


def test_6f8cd79b_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 8, 8, 8], [8, 0, 0, 8], [8, 0, 0, 8], [8, 0, 0, 8], [8, 8, 8, 8]]
)
    actual = solve_6f8cd79b(input_grid)
    assert actual == expected


def test_6f8cd79b_example_3():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 8, 8, 8, 8, 8],
 [8, 0, 0, 0, 0, 8],
 [8, 0, 0, 0, 0, 8],
 [8, 0, 0, 0, 0, 8],
 [8, 8, 8, 8, 8, 8]]
)
    actual = solve_6f8cd79b(input_grid)
    assert actual == expected



def test_6f8cd79b_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 8, 8, 8, 8, 8],
 [8, 0, 0, 0, 0, 8],
 [8, 0, 0, 0, 0, 8],
 [8, 0, 0, 0, 0, 8],
 [8, 0, 0, 0, 0, 8],
 [8, 0, 0, 0, 0, 8],
 [8, 8, 8, 8, 8, 8]]
    )
    actual = solve_6f8cd79b(input_grid)
    assert actual == expected

