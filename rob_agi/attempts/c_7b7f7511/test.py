import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_7b7f7511.main import solve_7b7f7511


def test_7b7f7511_example_0():
    input_grid = ColoredGrid(values=
[[1, 1, 3, 2, 1, 1, 3, 2],
 [1, 1, 3, 3, 1, 1, 3, 3],
 [3, 3, 1, 1, 3, 3, 1, 1],
 [2, 3, 1, 1, 2, 3, 1, 1]]
    )
    expected = ColoredGrid(values=
[[1, 1, 3, 2], [1, 1, 3, 3], [3, 3, 1, 1], [2, 3, 1, 1]]
)
    actual = solve_7b7f7511(input_grid)
    assert actual == expected


def test_7b7f7511_example_1():
    input_grid = ColoredGrid(values=
[[4, 4, 4, 4, 4, 4], [6, 4, 8, 6, 4, 8], [6, 6, 8, 6, 6, 8]]
    )
    expected = ColoredGrid(values=
[[4, 4, 4], [6, 4, 8], [6, 6, 8]]
)
    actual = solve_7b7f7511(input_grid)
    assert actual == expected


def test_7b7f7511_example_2():
    input_grid = ColoredGrid(values=
[[2, 3], [3, 2], [4, 4], [2, 3], [3, 2], [4, 4]]
    )
    expected = ColoredGrid(values=
[[2, 3], [3, 2], [4, 4]]
)
    actual = solve_7b7f7511(input_grid)
    assert actual == expected



def test_7b7f7511_test_case_0():
    input_grid = ColoredGrid(values=
[[5, 4, 5],
 [4, 5, 4],
 [6, 6, 4],
 [2, 6, 2],
 [5, 4, 5],
 [4, 5, 4],
 [6, 6, 4],
 [2, 6, 2]]
    )
    expected = ColoredGrid(values=
[[5, 4, 5], [4, 5, 4], [6, 6, 4], [2, 6, 2]]
    )
    actual = solve_7b7f7511(input_grid)
    assert actual == expected

