import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_e345f17b.main import solve_e345f17b

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_e345f17b_example_0():
    input_grid = ColoredGrid(values=
[[6, 6, 6, 6, 5, 0, 5, 0],
 [6, 0, 0, 0, 5, 5, 0, 0],
 [6, 0, 6, 6, 0, 0, 5, 5],
 [0, 0, 6, 0, 0, 5, 5, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0], [0, 0, 4, 4], [0, 4, 0, 0], [4, 0, 0, 4]]
)
    actual = solve_e345f17b(input_grid)
    assert actual == expected


def test_e345f17b_example_1():
    input_grid = ColoredGrid(values=
[[0, 6, 6, 0, 5, 5, 5, 0],
 [0, 6, 0, 6, 5, 0, 0, 5],
 [0, 6, 6, 6, 5, 5, 5, 5],
 [6, 0, 0, 0, 0, 5, 0, 5]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 4], [0, 0, 4, 0], [0, 0, 0, 0], [0, 0, 4, 0]]
)
    actual = solve_e345f17b(input_grid)
    assert actual == expected


def test_e345f17b_example_2():
    input_grid = ColoredGrid(values=
[[6, 6, 6, 0, 5, 0, 5, 5],
 [6, 0, 0, 0, 0, 5, 5, 5],
 [6, 0, 0, 0, 0, 0, 0, 0],
 [0, 6, 6, 6, 5, 5, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0], [0, 0, 0, 0], [0, 4, 4, 4], [0, 0, 0, 0]]
)
    actual = solve_e345f17b(input_grid)
    assert actual == expected


def test_e345f17b_example_3():
    input_grid = ColoredGrid(values=
[[6, 0, 6, 0, 0, 0, 5, 5],
 [0, 6, 6, 6, 5, 0, 5, 5],
 [6, 6, 0, 6, 5, 0, 5, 5],
 [6, 6, 0, 0, 5, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 4, 4]]
)
    actual = solve_e345f17b(input_grid)
    assert actual == expected



