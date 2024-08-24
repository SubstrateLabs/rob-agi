import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_66e6c45b.main import solve_66e6c45b


def test_66e6c45b_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0], [0, 3, 4, 0], [0, 7, 6, 0], [0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[3, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0], [7, 0, 0, 6]]
)
    actual = solve_66e6c45b(input_grid)
    assert actual == expected


def test_66e6c45b_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0], [0, 5, 6, 0], [0, 8, 3, 0], [0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[5, 0, 0, 6], [0, 0, 0, 0], [0, 0, 0, 0], [8, 0, 0, 3]]
)
    actual = solve_66e6c45b(input_grid)
    assert actual == expected



def test_66e6c45b_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0], [0, 2, 3, 0], [0, 4, 9, 0], [0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 9]]
    )
    actual = solve_66e6c45b(input_grid)
    assert actual == expected

