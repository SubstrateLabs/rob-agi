import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_67636eac.main import solve_67636eac

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_67636eac_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0],
 [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0],
 [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 2, 0, 0, 3, 0, 0, 8, 0],
 [2, 2, 2, 3, 3, 3, 8, 8, 8],
 [0, 2, 0, 0, 3, 0, 0, 8, 0]]
)
    actual = solve_67636eac(input_grid)
    assert actual == expected


def test_67636eac_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
 [0, 0, 3, 0, 3, 0, 0, 0, 0, 0],
 [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
 [0, 0, 0, 0, 8, 0, 8, 0, 0, 0],
 [0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 3, 0],
 [3, 0, 3],
 [0, 3, 0],
 [0, 1, 0],
 [1, 0, 1],
 [0, 1, 0],
 [0, 8, 0],
 [8, 0, 8],
 [0, 8, 0]]
)
    actual = solve_67636eac(input_grid)
    assert actual == expected


def test_67636eac_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 2, 0, 2, 0, 0, 0, 0],
 [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
 [0, 0, 0, 2, 0, 2, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[2, 0, 2], [0, 2, 0], [2, 0, 2], [1, 0, 1], [0, 1, 0], [1, 0, 1]]
)
    actual = solve_67636eac(input_grid)
    assert actual == expected



