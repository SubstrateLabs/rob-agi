import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_4852f2fa.main import solve_4852f2fa

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_4852f2fa_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 8, 8, 0, 0, 0, 0, 0],
 [0, 8, 8, 8, 0, 0, 4, 0, 0],
 [0, 0, 8, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 4, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 8, 8, 0, 8, 8], [8, 8, 8, 8, 8, 8], [0, 8, 0, 0, 8, 0]]
)
    actual = solve_4852f2fa(input_grid)
    assert actual == expected


def test_4852f2fa_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 4, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 8, 0, 0, 0, 0, 0, 0],
 [0, 8, 8, 0, 0, 0, 0, 0, 0],
 [0, 0, 8, 8, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 8, 0], [8, 8, 0], [0, 8, 8]]
)
    actual = solve_4852f2fa(input_grid)
    assert actual == expected


def test_4852f2fa_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 4, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 8, 8, 8, 0, 0, 0, 0, 0],
 [0, 0, 8, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 4, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 4, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0],
 [8, 8, 8, 8, 8, 8, 8, 8, 8],
 [0, 8, 0, 0, 8, 0, 0, 8, 0]]
)
    actual = solve_4852f2fa(input_grid)
    assert actual == expected


def test_4852f2fa_example_3():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0, 4, 0, 0, 0],
 [0, 0, 8, 0, 0, 0, 0, 0, 0],
 [8, 8, 0, 0, 0, 0, 0, 0, 0],
 [8, 8, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 4, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 4, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 8, 0, 0, 8, 0, 0, 8],
 [8, 8, 0, 8, 8, 0, 8, 8, 0],
 [8, 8, 0, 8, 8, 0, 8, 8, 0]]
)
    actual = solve_4852f2fa(input_grid)
    assert actual == expected


def test_4852f2fa_example_4():
    input_grid = ColoredGrid(values=
[[0, 8, 8, 0, 0, 0, 0, 0, 0],
 [8, 8, 0, 0, 4, 0, 0, 0, 0],
 [0, 8, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 4, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 4, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 4, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 8, 8, 0, 8, 8, 0, 8, 8, 0, 8, 8],
 [8, 8, 0, 8, 8, 0, 8, 8, 0, 8, 8, 0],
 [0, 8, 0, 0, 8, 0, 0, 8, 0, 0, 8, 0]]
)
    actual = solve_4852f2fa(input_grid)
    assert actual == expected



