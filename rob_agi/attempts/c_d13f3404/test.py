import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_d13f3404.main import solve_d13f3404


def test_d13f3404_example_0():
    input_grid = ColoredGrid(values=
[[6, 1, 0], [3, 0, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[6, 1, 0, 0, 0, 0],
 [3, 6, 1, 0, 0, 0],
 [0, 3, 6, 1, 0, 0],
 [0, 0, 3, 6, 1, 0],
 [0, 0, 0, 3, 6, 1],
 [0, 0, 0, 0, 3, 6]]
)
    actual = solve_d13f3404(input_grid)
    assert actual == expected


def test_d13f3404_example_1():
    input_grid = ColoredGrid(values=
[[0, 4, 0], [0, 8, 0], [2, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 4, 0, 0, 0, 0],
 [0, 8, 4, 0, 0, 0],
 [2, 0, 8, 4, 0, 0],
 [0, 2, 0, 8, 4, 0],
 [0, 0, 2, 0, 8, 4],
 [0, 0, 0, 2, 0, 8]]
)
    actual = solve_d13f3404(input_grid)
    assert actual == expected


def test_d13f3404_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 6], [1, 3, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 6, 0, 0, 0],
 [1, 3, 0, 6, 0, 0],
 [0, 1, 3, 0, 6, 0],
 [0, 0, 1, 3, 0, 6],
 [0, 0, 0, 1, 3, 0],
 [0, 0, 0, 0, 1, 3]]
)
    actual = solve_d13f3404(input_grid)
    assert actual == expected



def test_d13f3404_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 3], [0, 0, 0], [0, 4, 9]]
    )
    expected = ColoredGrid(values=
[[0, 0, 3, 0, 0, 0],
 [0, 0, 0, 3, 0, 0],
 [0, 4, 9, 0, 3, 0],
 [0, 0, 4, 9, 0, 3],
 [0, 0, 0, 4, 9, 0],
 [0, 0, 0, 0, 4, 9]]
    )
    actual = solve_d13f3404(input_grid)
    assert actual == expected

