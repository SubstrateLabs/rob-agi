import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_1e0a9b12.main import solve_1e0a9b12


def test_1e0a9b12_example_0():
    input_grid = ColoredGrid(values=
[[0, 4, 0, 9], [0, 0, 0, 0], [0, 4, 6, 0], [1, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0], [0, 0, 0, 0], [0, 4, 0, 0], [1, 4, 6, 9]]
)
    actual = solve_1e0a9b12(input_grid)
    assert actual == expected


def test_1e0a9b12_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 9],
 [0, 0, 0, 8, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [4, 0, 0, 0, 0, 0],
 [4, 0, 7, 8, 0, 0],
 [4, 0, 7, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [4, 0, 0, 0, 0, 0],
 [4, 0, 7, 8, 0, 0],
 [4, 0, 7, 8, 0, 9]]
)
    actual = solve_1e0a9b12(input_grid)
    assert actual == expected


def test_1e0a9b12_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 1, 0],
 [0, 3, 0, 0, 0],
 [0, 3, 0, 1, 2],
 [6, 0, 0, 0, 0],
 [0, 3, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 3, 0, 0, 0],
 [0, 3, 0, 1, 0],
 [6, 3, 0, 1, 2]]
)
    actual = solve_1e0a9b12(input_grid)
    assert actual == expected



def test_1e0a9b12_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 2, 0, 4, 3],
 [5, 0, 0, 0, 0],
 [0, 0, 6, 0, 0],
 [5, 2, 0, 4, 0],
 [5, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [5, 0, 0, 0, 0],
 [5, 2, 0, 4, 0],
 [5, 2, 6, 4, 3]]
    )
    actual = solve_1e0a9b12(input_grid)
    assert actual == expected

