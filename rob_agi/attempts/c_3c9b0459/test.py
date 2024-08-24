import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_3c9b0459.main import solve_3c9b0459


def test_3c9b0459_example_0():
    input_grid = ColoredGrid(values=
[[2, 2, 1], [2, 1, 2], [2, 8, 1]]
    )
    expected = ColoredGrid(values=
[[1, 8, 2], [2, 1, 2], [1, 2, 2]]
)
    actual = solve_3c9b0459(input_grid)
    assert actual == expected


def test_3c9b0459_example_1():
    input_grid = ColoredGrid(values=
[[9, 2, 4], [2, 4, 4], [2, 9, 2]]
    )
    expected = ColoredGrid(values=
[[2, 9, 2], [4, 4, 2], [4, 2, 9]]
)
    actual = solve_3c9b0459(input_grid)
    assert actual == expected


def test_3c9b0459_example_2():
    input_grid = ColoredGrid(values=
[[8, 8, 8], [5, 5, 8], [8, 5, 5]]
    )
    expected = ColoredGrid(values=
[[5, 5, 8], [8, 5, 5], [8, 8, 8]]
)
    actual = solve_3c9b0459(input_grid)
    assert actual == expected


def test_3c9b0459_example_3():
    input_grid = ColoredGrid(values=
[[3, 2, 9], [9, 9, 9], [2, 3, 3]]
    )
    expected = ColoredGrid(values=
[[3, 3, 2], [9, 9, 9], [9, 2, 3]]
)
    actual = solve_3c9b0459(input_grid)
    assert actual == expected



def test_3c9b0459_test_case_0():
    input_grid = ColoredGrid(values=
[[6, 4, 4], [6, 6, 4], [4, 6, 7]]
    )
    expected = ColoredGrid(values=
[[7, 6, 4], [4, 6, 6], [4, 4, 6]]
    )
    actual = solve_3c9b0459(input_grid)
    assert actual == expected

