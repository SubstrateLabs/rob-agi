import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_3bd67248.main import solve_3bd67248

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_3bd67248_example_0():
    input_grid = ColoredGrid(values=
[[6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
 [6, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]
)
    actual = solve_3bd67248(input_grid)
    assert actual == expected


def test_3bd67248_example_1():
    input_grid = ColoredGrid(values=
[[5, 0, 0], [5, 0, 0], [5, 0, 0]]
    )
    expected = ColoredGrid(values=
[[5, 0, 2], [5, 2, 0], [5, 4, 4]]
)
    actual = solve_3bd67248(input_grid)
    assert actual == expected


def test_3bd67248_example_2():
    input_grid = ColoredGrid(values=
[[8, 0, 0, 0, 0, 0, 0],
 [8, 0, 0, 0, 0, 0, 0],
 [8, 0, 0, 0, 0, 0, 0],
 [8, 0, 0, 0, 0, 0, 0],
 [8, 0, 0, 0, 0, 0, 0],
 [8, 0, 0, 0, 0, 0, 0],
 [8, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 0, 0, 0, 0, 0, 2],
 [8, 0, 0, 0, 0, 2, 0],
 [8, 0, 0, 0, 2, 0, 0],
 [8, 0, 0, 2, 0, 0, 0],
 [8, 0, 2, 0, 0, 0, 0],
 [8, 2, 0, 0, 0, 0, 0],
 [8, 4, 4, 4, 4, 4, 4]]
)
    actual = solve_3bd67248(input_grid)
    assert actual == expected



