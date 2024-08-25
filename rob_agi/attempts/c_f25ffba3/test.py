import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_f25ffba3.main import solve_f25ffba3


def test_f25ffba3_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 9],
 [0, 0, 3, 9],
 [0, 0, 3, 9],
 [2, 0, 3, 9],
 [2, 4, 3, 9]]
    )
    expected = ColoredGrid(values=
[[2, 4, 3, 9],
 [2, 0, 3, 9],
 [0, 0, 3, 9],
 [0, 0, 3, 9],
 [0, 0, 0, 9],
 [0, 0, 0, 9],
 [0, 0, 3, 9],
 [0, 0, 3, 9],
 [2, 0, 3, 9],
 [2, 4, 3, 9]]
)
    actual = solve_f25ffba3(input_grid)
    assert actual == expected


def test_f25ffba3_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 2],
 [0, 0, 0, 2],
 [0, 8, 0, 2],
 [0, 3, 8, 2],
 [3, 3, 8, 2]]
    )
    expected = ColoredGrid(values=
[[3, 3, 8, 2],
 [0, 3, 8, 2],
 [0, 8, 0, 2],
 [0, 0, 0, 2],
 [0, 0, 0, 2],
 [0, 0, 0, 2],
 [0, 0, 0, 2],
 [0, 8, 0, 2],
 [0, 3, 8, 2],
 [3, 3, 8, 2]]
)
    actual = solve_f25ffba3(input_grid)
    assert actual == expected



def test_f25ffba3_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 1, 0, 0],
 [7, 1, 0, 0],
 [7, 1, 3, 0],
 [7, 1, 3, 3],
 [7, 1, 4, 4]]
    )
    expected = ColoredGrid(values=
[[7, 1, 4, 4],
 [7, 1, 3, 3],
 [7, 1, 3, 0],
 [7, 1, 0, 0],
 [0, 1, 0, 0],
 [0, 1, 0, 0],
 [7, 1, 0, 0],
 [7, 1, 3, 0],
 [7, 1, 3, 3],
 [7, 1, 4, 4]]
    )
    actual = solve_f25ffba3(input_grid)
    assert actual == expected

