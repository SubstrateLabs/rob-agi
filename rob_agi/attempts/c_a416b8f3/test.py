import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_a416b8f3.main import solve_a416b8f3


def test_a416b8f3_example_0():
    input_grid = ColoredGrid(values=
[[0, 5, 0], [5, 5, 2], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 5, 0, 0, 5, 0], [5, 5, 2, 5, 5, 2], [0, 0, 0, 0, 0, 0]]
)
    actual = solve_a416b8f3(input_grid)
    assert actual == expected


def test_a416b8f3_example_1():
    input_grid = ColoredGrid(values=
[[3, 0, 0], [2, 3, 0], [2, 1, 8], [0, 1, 0]]
    )
    expected = ColoredGrid(values=
[[3, 0, 0, 3, 0, 0], [2, 3, 0, 2, 3, 0], [2, 1, 8, 2, 1, 8], [0, 1, 0, 0, 1, 0]]
)
    actual = solve_a416b8f3(input_grid)
    assert actual == expected


def test_a416b8f3_example_2():
    input_grid = ColoredGrid(values=
[[5, 2, 3, 0], [2, 5, 3, 0], [5, 2, 8, 8], [0, 0, 6, 0]]
    )
    expected = ColoredGrid(values=
[[5, 2, 3, 0, 5, 2, 3, 0],
 [2, 5, 3, 0, 2, 5, 3, 0],
 [5, 2, 8, 8, 5, 2, 8, 8],
 [0, 0, 6, 0, 0, 0, 6, 0]]
)
    actual = solve_a416b8f3(input_grid)
    assert actual == expected



def test_a416b8f3_test_case_0():
    input_grid = ColoredGrid(values=
[[4, 0, 0, 0], [4, 5, 0, 0], [0, 5, 6, 0], [6, 6, 1, 0], [0, 0, 0, 1]]
    )
    expected = ColoredGrid(values=
[[4, 0, 0, 0, 4, 0, 0, 0],
 [4, 5, 0, 0, 4, 5, 0, 0],
 [0, 5, 6, 0, 0, 5, 6, 0],
 [6, 6, 1, 0, 6, 6, 1, 0],
 [0, 0, 0, 1, 0, 0, 0, 1]]
    )
    actual = solve_a416b8f3(input_grid)
    assert actual == expected

