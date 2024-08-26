import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_d19f7514.main import solve_d19f7514

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_d19f7514_example_0():
    input_grid = ColoredGrid(values=
[[0, 3, 3, 3],
 [0, 3, 0, 3],
 [0, 0, 0, 0],
 [3, 0, 3, 3],
 [3, 0, 0, 0],
 [0, 3, 0, 3],
 [0, 5, 0, 5],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [5, 0, 5, 0],
 [5, 0, 0, 0],
 [5, 5, 0, 5]]
    )
    expected = ColoredGrid(values=
[[0, 4, 4, 4],
 [0, 4, 0, 4],
 [0, 0, 0, 0],
 [4, 0, 4, 4],
 [4, 0, 0, 0],
 [4, 4, 0, 4]]
)
    actual = solve_d19f7514(input_grid)
    assert actual == expected


def test_d19f7514_example_1():
    input_grid = ColoredGrid(values=
[[3, 3, 0, 3],
 [3, 0, 3, 3],
 [0, 3, 0, 0],
 [0, 0, 3, 0],
 [3, 0, 3, 0],
 [0, 0, 0, 3],
 [5, 0, 0, 0],
 [0, 5, 5, 5],
 [5, 0, 0, 5],
 [0, 5, 5, 5],
 [5, 5, 5, 0],
 [5, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[4, 4, 0, 4],
 [4, 4, 4, 4],
 [4, 4, 0, 4],
 [0, 4, 4, 4],
 [4, 4, 4, 0],
 [4, 0, 0, 4]]
)
    actual = solve_d19f7514(input_grid)
    assert actual == expected


def test_d19f7514_example_2():
    input_grid = ColoredGrid(values=
[[3, 3, 0, 0],
 [3, 0, 0, 0],
 [0, 0, 0, 3],
 [0, 0, 3, 3],
 [3, 0, 0, 0],
 [3, 3, 3, 3],
 [0, 5, 0, 0],
 [5, 5, 0, 0],
 [5, 0, 5, 0],
 [5, 5, 5, 5],
 [5, 5, 5, 0],
 [5, 0, 5, 0]]
    )
    expected = ColoredGrid(values=
[[4, 4, 0, 0],
 [4, 4, 0, 0],
 [4, 0, 4, 4],
 [4, 4, 4, 4],
 [4, 4, 4, 0],
 [4, 4, 4, 4]]
)
    actual = solve_d19f7514(input_grid)
    assert actual == expected


def test_d19f7514_example_3():
    input_grid = ColoredGrid(values=
[[3, 3, 0, 0],
 [0, 3, 3, 3],
 [3, 3, 0, 3],
 [0, 3, 3, 0],
 [3, 0, 3, 0],
 [3, 0, 0, 0],
 [0, 5, 5, 5],
 [5, 5, 5, 5],
 [5, 5, 5, 0],
 [5, 5, 5, 5],
 [5, 0, 0, 0],
 [0, 5, 5, 0]]
    )
    expected = ColoredGrid(values=
[[4, 4, 4, 4],
 [4, 4, 4, 4],
 [4, 4, 4, 4],
 [4, 4, 4, 4],
 [4, 0, 4, 0],
 [4, 4, 4, 0]]
)
    actual = solve_d19f7514(input_grid)
    assert actual == expected



