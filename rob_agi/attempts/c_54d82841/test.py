import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_54d82841.main import solve_54d82841


def test_54d82841_example_0():
    input_grid = ColoredGrid(values=
[[0, 6, 6, 6, 0, 0, 0, 0],
 [0, 6, 0, 6, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 6, 6, 6],
 [0, 0, 0, 0, 0, 6, 0, 6],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 6, 6, 6, 0, 0, 0, 0],
 [0, 6, 0, 6, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 6, 6, 6],
 [0, 0, 0, 0, 0, 6, 0, 6],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 4, 0, 0, 0, 4, 0]]
)
    actual = solve_54d82841(input_grid)
    assert actual == expected


def test_54d82841_example_1():
    input_grid = ColoredGrid(values=
[[0, 3, 3, 3, 0],
 [0, 3, 0, 3, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 3, 3, 3, 0],
 [0, 3, 0, 3, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 4, 0, 0]]
)
    actual = solve_54d82841(input_grid)
    assert actual == expected


def test_54d82841_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0],
 [0, 8, 8, 8, 0, 0, 0],
 [0, 8, 0, 8, 6, 6, 6],
 [0, 0, 0, 0, 6, 0, 6],
 [0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0],
 [0, 8, 8, 8, 0, 0, 0],
 [0, 8, 0, 8, 6, 6, 6],
 [0, 0, 0, 0, 6, 0, 6],
 [0, 0, 4, 0, 0, 4, 0]]
)
    actual = solve_54d82841(input_grid)
    assert actual == expected



def test_54d82841_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0],
 [0, 5, 0, 5, 0, 8, 8, 8, 0, 0, 0],
 [0, 0, 0, 0, 0, 8, 0, 8, 3, 3, 3],
 [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0],
 [0, 5, 0, 5, 0, 8, 8, 8, 0, 0, 0],
 [0, 0, 0, 0, 0, 8, 0, 8, 3, 3, 3],
 [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 4, 0, 0, 0, 4, 0, 0, 4, 0]]
    )
    actual = solve_54d82841(input_grid)
    assert actual == expected

