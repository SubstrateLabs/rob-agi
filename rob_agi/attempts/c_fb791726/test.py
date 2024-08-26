import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_fb791726.main import solve_fb791726

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_fb791726_example_0():
    input_grid = ColoredGrid(values=
[[0, 4, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 4, 0, 0, 0, 0],
 [0, 0, 0, 0, 4, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 4, 0]]
    )
    expected = ColoredGrid(values=
[[0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
 [0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
 [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
 [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
 [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
 [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
 [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0]]
)
    actual = solve_fb791726(input_grid)
    assert actual == expected


def test_fb791726_example_1():
    input_grid = ColoredGrid(values=
[[0, 8, 0], [0, 0, 0], [0, 8, 0]]
    )
    expected = ColoredGrid(values=
[[0, 8, 0, 0, 0, 0],
 [3, 3, 3, 3, 3, 3],
 [0, 8, 0, 0, 0, 0],
 [0, 0, 0, 0, 8, 0],
 [3, 3, 3, 3, 3, 3],
 [0, 0, 0, 0, 8, 0]]
)
    actual = solve_fb791726(input_grid)
    assert actual == expected


def test_fb791726_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 7, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 7, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
 [0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
 [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
    actual = solve_fb791726(input_grid)
    assert actual == expected



