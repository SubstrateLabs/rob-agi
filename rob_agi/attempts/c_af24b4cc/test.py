import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_af24b4cc.main import solve_af24b4cc


def test_af24b4cc_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 3, 3, 0, 6, 6, 0, 9, 7, 0],
 [0, 8, 3, 0, 6, 3, 0, 9, 7, 0],
 [0, 3, 8, 0, 3, 6, 0, 7, 7, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 3, 3, 0, 2, 2, 0, 6, 1, 0],
 [0, 2, 3, 0, 5, 5, 0, 1, 1, 0],
 [0, 2, 3, 0, 5, 5, 0, 1, 6, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0], [0, 3, 6, 7, 0], [0, 3, 5, 1, 0], [0, 0, 0, 0, 0]]
)
    actual = solve_af24b4cc(input_grid)
    assert actual == expected


def test_af24b4cc_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 1, 0, 5, 5, 0, 4, 4, 0],
 [0, 1, 1, 0, 3, 3, 0, 4, 4, 0],
 [0, 3, 3, 0, 5, 5, 0, 4, 8, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 2, 2, 0, 7, 1, 0, 9, 9, 0],
 [0, 2, 2, 0, 7, 7, 0, 1, 9, 0],
 [0, 2, 2, 0, 7, 1, 0, 9, 9, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0], [0, 1, 5, 4, 0], [0, 2, 7, 9, 0], [0, 0, 0, 0, 0]]
)
    actual = solve_af24b4cc(input_grid)
    assert actual == expected


def test_af24b4cc_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 3, 5, 0, 8, 4, 0, 7, 7, 0],
 [0, 5, 3, 0, 8, 8, 0, 7, 6, 0],
 [0, 3, 3, 0, 8, 4, 0, 6, 7, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 3, 3, 0, 2, 2, 0, 1, 3, 0],
 [0, 4, 3, 0, 2, 2, 0, 1, 1, 0],
 [0, 3, 3, 0, 1, 2, 0, 1, 3, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0], [0, 3, 8, 7, 0], [0, 3, 2, 1, 0], [0, 0, 0, 0, 0]]
)
    actual = solve_af24b4cc(input_grid)
    assert actual == expected



def test_af24b4cc_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 1, 0, 3, 3, 0, 4, 4, 0],
 [0, 3, 1, 0, 8, 3, 0, 4, 4, 0],
 [0, 1, 1, 0, 3, 8, 0, 8, 4, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 2, 2, 0, 3, 5, 0, 2, 2, 0],
 [0, 6, 6, 0, 5, 5, 0, 2, 2, 0],
 [0, 2, 2, 0, 5, 3, 0, 2, 2, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0], [0, 1, 3, 4, 0], [0, 2, 5, 2, 0], [0, 0, 0, 0, 0]]
    )
    actual = solve_af24b4cc(input_grid)
    assert actual == expected

