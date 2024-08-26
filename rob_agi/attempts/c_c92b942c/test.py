import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_c92b942c.main import solve_c92b942c

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_c92b942c_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0], [0, 6, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[3, 0, 0, 3, 0, 0, 3, 0, 0],
 [1, 6, 1, 1, 6, 1, 1, 6, 1],
 [0, 0, 3, 0, 0, 3, 0, 0, 3],
 [3, 0, 0, 3, 0, 0, 3, 0, 0],
 [1, 6, 1, 1, 6, 1, 1, 6, 1],
 [0, 0, 3, 0, 0, 3, 0, 0, 3],
 [3, 0, 0, 3, 0, 0, 3, 0, 0],
 [1, 6, 1, 1, 6, 1, 1, 6, 1],
 [0, 0, 3, 0, 0, 3, 0, 0, 3]]
)
    actual = solve_c92b942c(input_grid)
    assert actual == expected


def test_c92b942c_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 5, 0], [0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[1, 1, 5, 1, 1, 1, 5, 1, 1, 1, 5, 1],
 [0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3],
 [1, 1, 5, 1, 1, 1, 5, 1, 1, 1, 5, 1],
 [0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3],
 [1, 1, 5, 1, 1, 1, 5, 1, 1, 1, 5, 1],
 [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3]]
)
    actual = solve_c92b942c(input_grid)
    assert actual == expected


def test_c92b942c_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0], [0, 0, 4, 0, 0, 0], [0, 0, 0, 0, 0, 0], [4, 0, 0, 0, 4, 0]]
    )
    expected = ColoredGrid(values=
[[0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
 [1, 1, 4, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 4, 1, 1, 1],
 [0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0],
 [4, 1, 1, 1, 4, 1, 4, 1, 1, 1, 4, 1, 4, 1, 1, 1, 4, 1],
 [0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3],
 [1, 1, 4, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 4, 1, 1, 1],
 [0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0],
 [4, 1, 1, 1, 4, 1, 4, 1, 1, 1, 4, 1, 4, 1, 1, 1, 4, 1],
 [0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3],
 [1, 1, 4, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 4, 1, 1, 1],
 [0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0],
 [4, 1, 1, 1, 4, 1, 4, 1, 1, 1, 4, 1, 4, 1, 1, 1, 4, 1]]
)
    actual = solve_c92b942c(input_grid)
    assert actual == expected


def test_c92b942c_example_3():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 2, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0],
 [1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1],
 [0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0],
 [1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1],
 [0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0],
 [1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1],
 [0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_c92b942c(input_grid)
    assert actual == expected



