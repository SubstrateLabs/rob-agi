import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_a740d043.main import solve_a740d043


def test_a740d043_example_0():
    input_grid = ColoredGrid(values=
[[1, 1, 1, 1, 1, 1, 1],
 [1, 2, 2, 1, 1, 1, 1],
 [1, 2, 2, 3, 1, 1, 1],
 [1, 1, 1, 2, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1]]
    )
    expected = ColoredGrid(values=
[[2, 2, 0], [2, 2, 3], [0, 0, 2]]
)
    actual = solve_a740d043(input_grid)
    assert actual == expected


def test_a740d043_example_1():
    input_grid = ColoredGrid(values=
[[1, 1, 1, 1, 1, 1, 1],
 [1, 1, 3, 1, 2, 1, 1],
 [1, 1, 3, 1, 2, 1, 1],
 [1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1]]
    )
    expected = ColoredGrid(values=
[[3, 0, 2], [3, 0, 2]]
)
    actual = solve_a740d043(input_grid)
    assert actual == expected


def test_a740d043_example_2():
    input_grid = ColoredGrid(values=
[[1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1],
 [1, 5, 5, 1, 1, 1],
 [1, 5, 5, 1, 1, 1],
 [1, 6, 6, 1, 1, 1],
 [1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1]]
    )
    expected = ColoredGrid(values=
[[5, 5], [5, 5], [6, 6]]
)
    actual = solve_a740d043(input_grid)
    assert actual == expected



def test_a740d043_test_case_0():
    input_grid = ColoredGrid(values=
[[1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1],
 [1, 1, 1, 2, 1, 1],
 [1, 1, 2, 3, 1, 1],
 [1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1]]
    )
    expected = ColoredGrid(values=
[[0, 2], [2, 3]]
    )
    actual = solve_a740d043(input_grid)
    assert actual == expected

