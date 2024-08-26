import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_ca8de6ea.main import solve_ca8de6ea

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_ca8de6ea_example_0():
    input_grid = ColoredGrid(values=
[[1, 0, 0, 0, 9],
 [0, 5, 0, 8, 0],
 [0, 0, 7, 0, 0],
 [0, 8, 0, 5, 0],
 [9, 0, 0, 0, 1]]
    )
    expected = ColoredGrid(values=
[[1, 5, 9], [8, 7, 8], [9, 5, 1]]
)
    actual = solve_ca8de6ea(input_grid)
    assert actual == expected


def test_ca8de6ea_example_1():
    input_grid = ColoredGrid(values=
[[6, 0, 0, 0, 7],
 [0, 2, 0, 4, 0],
 [0, 0, 3, 0, 0],
 [0, 4, 0, 2, 0],
 [7, 0, 0, 0, 6]]
    )
    expected = ColoredGrid(values=
[[6, 2, 7], [4, 3, 4], [7, 2, 6]]
)
    actual = solve_ca8de6ea(input_grid)
    assert actual == expected


def test_ca8de6ea_example_2():
    input_grid = ColoredGrid(values=
[[2, 0, 0, 0, 1],
 [0, 3, 0, 6, 0],
 [0, 0, 4, 0, 0],
 [0, 6, 0, 3, 0],
 [1, 0, 0, 0, 2]]
    )
    expected = ColoredGrid(values=
[[2, 3, 1], [6, 4, 6], [1, 3, 2]]
)
    actual = solve_ca8de6ea(input_grid)
    assert actual == expected



