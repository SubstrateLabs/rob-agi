import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_d511f180.main import solve_d511f180


def test_d511f180_example_0():
    input_grid = ColoredGrid(values=
[[2, 7, 8, 8, 8],
 [5, 5, 6, 5, 4],
 [8, 5, 5, 5, 2],
 [8, 8, 4, 3, 6],
 [6, 5, 1, 9, 3]]
    )
    expected = ColoredGrid(values=
[[2, 7, 5, 5, 5],
 [8, 8, 6, 8, 4],
 [5, 8, 8, 8, 2],
 [5, 5, 4, 3, 6],
 [6, 8, 1, 9, 3]]
)
    actual = solve_d511f180(input_grid)
    assert actual == expected


def test_d511f180_example_1():
    input_grid = ColoredGrid(values=
[[3, 5, 1], [4, 5, 8], [2, 4, 9]]
    )
    expected = ColoredGrid(values=
[[3, 8, 1], [4, 8, 5], [2, 4, 9]]
)
    actual = solve_d511f180(input_grid)
    assert actual == expected


def test_d511f180_example_2():
    input_grid = ColoredGrid(values=
[[6, 5, 3], [5, 7, 5], [8, 8, 2]]
    )
    expected = ColoredGrid(values=
[[6, 8, 3], [8, 7, 8], [5, 5, 2]]
)
    actual = solve_d511f180(input_grid)
    assert actual == expected



def test_d511f180_test_case_0():
    input_grid = ColoredGrid(values=
[[8, 8, 4, 5], [3, 8, 7, 5], [3, 7, 1, 9], [6, 4, 8, 8]]
    )
    expected = ColoredGrid(values=
[[5, 5, 4, 8], [3, 5, 7, 8], [3, 7, 1, 9], [6, 4, 5, 5]]
    )
    actual = solve_d511f180(input_grid)
    assert actual == expected

