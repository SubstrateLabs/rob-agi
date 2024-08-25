import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_aabf363d.main import solve_aabf363d


def test_aabf363d_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0],
 [0, 2, 2, 2, 0, 0, 0],
 [0, 0, 2, 0, 0, 0, 0],
 [0, 2, 2, 2, 2, 0, 0],
 [0, 0, 2, 2, 2, 0, 0],
 [0, 0, 0, 2, 0, 0, 0],
 [4, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0],
 [0, 4, 4, 4, 0, 0, 0],
 [0, 0, 4, 0, 0, 0, 0],
 [0, 4, 4, 4, 4, 0, 0],
 [0, 0, 4, 4, 4, 0, 0],
 [0, 0, 0, 4, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_aabf363d(input_grid)
    assert actual == expected


def test_aabf363d_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 3, 0, 0, 0],
 [0, 0, 3, 3, 3, 0, 0],
 [0, 3, 3, 3, 3, 0, 0],
 [0, 3, 3, 0, 0, 0, 0],
 [0, 0, 3, 3, 0, 0, 0],
 [6, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 6, 0, 0, 0],
 [0, 0, 6, 6, 6, 0, 0],
 [0, 6, 6, 6, 6, 0, 0],
 [0, 6, 6, 0, 0, 0, 0],
 [0, 0, 6, 6, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_aabf363d(input_grid)
    assert actual == expected



def test_aabf363d_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0],
 [0, 8, 8, 8, 0, 0, 0],
 [0, 8, 8, 8, 8, 8, 0],
 [0, 0, 0, 8, 8, 0, 0],
 [0, 0, 8, 8, 0, 0, 0],
 [0, 0, 8, 8, 8, 0, 0],
 [2, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0],
 [0, 2, 2, 2, 0, 0, 0],
 [0, 2, 2, 2, 2, 2, 0],
 [0, 0, 0, 2, 2, 0, 0],
 [0, 0, 2, 2, 0, 0, 0],
 [0, 0, 2, 2, 2, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
    )
    actual = solve_aabf363d(input_grid)
    assert actual == expected

