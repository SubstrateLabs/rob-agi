import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_59341089.main import solve_59341089

# This file is important because it contains assertions that represent the example and test cases.
# These assertions are used both to verify the correctness of the implementation
# and to illustrate the problem itself.


def test_59341089_example_0():
    input_grid = ColoredGrid(values=
[[7, 5, 7], [5, 5, 7], [7, 7, 5]]
    )
    expected = ColoredGrid(values=
[[7, 5, 7, 7, 5, 7, 7, 5, 7, 7, 5, 7],
 [7, 5, 5, 5, 5, 7, 7, 5, 5, 5, 5, 7],
 [5, 7, 7, 7, 7, 5, 5, 7, 7, 7, 7, 5]]
)
    actual = solve_59341089(input_grid)
    assert actual == expected


def test_59341089_example_1():
    input_grid = ColoredGrid(values=
[[7, 7, 8], [5, 8, 8], [5, 8, 8]]
    )
    expected = ColoredGrid(values=
[[8, 7, 7, 7, 7, 8, 8, 7, 7, 7, 7, 8],
 [8, 8, 5, 5, 8, 8, 8, 8, 5, 5, 8, 8],
 [8, 8, 5, 5, 8, 8, 8, 8, 5, 5, 8, 8]]
)
    actual = solve_59341089(input_grid)
    assert actual == expected


def test_59341089_example_2():
    input_grid = ColoredGrid(values=
[[8, 8, 8], [5, 5, 7], [5, 7, 8]]
    )
    expected = ColoredGrid(values=
[[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [7, 5, 5, 5, 5, 7, 7, 5, 5, 5, 5, 7],
 [8, 7, 5, 5, 7, 8, 8, 7, 5, 5, 7, 8]]
)
    actual = solve_59341089(input_grid)
    assert actual == expected


def test_59341089_example_3():
    input_grid = ColoredGrid(values=
[[8, 8, 7], [7, 5, 5], [5, 7, 8]]
    )
    expected = ColoredGrid(values=
[[7, 8, 8, 8, 8, 7, 7, 8, 8, 8, 8, 7],
 [5, 5, 7, 7, 5, 5, 5, 5, 7, 7, 5, 5],
 [8, 7, 5, 5, 7, 8, 8, 7, 5, 5, 7, 8]]
)
    actual = solve_59341089(input_grid)
    assert actual == expected



