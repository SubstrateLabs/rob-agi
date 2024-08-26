import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_ae58858e.main import solve_ae58858e

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_ae58858e_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0],
 [2, 2, 0, 0, 0, 2, 2, 0],
 [0, 2, 2, 0, 0, 2, 2, 0],
 [0, 0, 0, 0, 0, 0, 2, 2],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 2, 2, 2, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 2, 0, 0],
 [0, 0, 2, 2, 0, 0, 0, 0],
 [2, 0, 2, 2, 0, 0, 2, 2],
 [2, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0],
 [6, 6, 0, 0, 0, 6, 6, 0],
 [0, 6, 6, 0, 0, 6, 6, 0],
 [0, 0, 0, 0, 0, 0, 6, 6],
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, 2, 2, 2, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 2, 0, 0],
 [0, 0, 6, 6, 0, 0, 0, 0],
 [2, 0, 6, 6, 0, 0, 2, 2],
 [2, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_ae58858e(input_grid)
    assert actual == expected


def test_ae58858e_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [2, 2, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0],
 [0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 0],
 [0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 2, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
 [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
 [0, 2, 0, 0, 2, 2, 0, 0, 0, 2, 2, 2],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
 [0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0],
 [0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [6, 6, 6, 0, 0, 0, 6, 0, 0, 0, 0, 0],
 [0, 6, 6, 0, 0, 0, 6, 6, 0, 0, 0, 0],
 [0, 6, 6, 6, 0, 0, 6, 6, 0, 0, 2, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
 [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
 [0, 2, 0, 0, 2, 2, 0, 0, 0, 6, 6, 6],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6],
 [0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 6, 0],
 [0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0]]
)
    actual = solve_ae58858e(input_grid)
    assert actual == expected


def test_ae58858e_example_2():
    input_grid = ColoredGrid(values=
[[2, 2, 0, 0, 0, 2],
 [2, 2, 0, 0, 0, 2],
 [0, 0, 0, 2, 0, 0],
 [0, 2, 0, 0, 0, 0],
 [0, 0, 0, 2, 0, 2],
 [0, 2, 2, 2, 0, 0]]
    )
    expected = ColoredGrid(values=
[[6, 6, 0, 0, 0, 2],
 [6, 6, 0, 0, 0, 2],
 [0, 0, 0, 2, 0, 0],
 [0, 2, 0, 0, 0, 0],
 [0, 0, 0, 6, 0, 2],
 [0, 6, 6, 6, 0, 0]]
)
    actual = solve_ae58858e(input_grid)
    assert actual == expected


def test_ae58858e_example_3():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 2, 2, 0, 0, 0, 0, 0, 0],
 [0, 0, 2, 0, 0, 0, 2, 2, 0],
 [0, 0, 0, 0, 0, 2, 2, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 2, 0, 0, 0, 0, 0],
 [0, 2, 2, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 2, 0],
 [0, 0, 0, 0, 0, 0, 0, 2, 0],
 [0, 0, 0, 2, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 2, 2, 0, 0, 0, 0, 0, 0],
 [0, 0, 2, 0, 0, 0, 6, 6, 0],
 [0, 0, 0, 0, 0, 6, 6, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 2, 0, 0, 0, 0, 0],
 [0, 2, 2, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 2, 0],
 [0, 0, 0, 0, 0, 0, 0, 2, 0],
 [0, 0, 0, 2, 0, 0, 0, 0, 0]]
)
    actual = solve_ae58858e(input_grid)
    assert actual == expected



