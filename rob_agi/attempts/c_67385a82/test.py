import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_67385a82.main import solve_67385a82


def test_67385a82_example_0():
    input_grid = ColoredGrid(values=
[[3, 3, 0], [0, 3, 0], [3, 0, 3]]
    )
    expected = ColoredGrid(values=
[[8, 8, 0], [0, 8, 0], [3, 0, 3]]
)
    actual = solve_67385a82(input_grid)
    assert actual == expected


def test_67385a82_example_1():
    input_grid = ColoredGrid(values=
[[0, 3, 0, 0, 0, 3], [0, 3, 3, 3, 0, 0], [0, 0, 0, 0, 3, 0], [0, 3, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 8, 0, 0, 0, 3], [0, 8, 8, 8, 0, 0], [0, 0, 0, 0, 3, 0], [0, 3, 0, 0, 0, 0]]
)
    actual = solve_67385a82(input_grid)
    assert actual == expected


def test_67385a82_example_2():
    input_grid = ColoredGrid(values=
[[3, 3, 0, 3], [3, 3, 0, 0], [3, 0, 0, 3], [0, 0, 3, 3]]
    )
    expected = ColoredGrid(values=
[[8, 8, 0, 3], [8, 8, 0, 0], [8, 0, 0, 8], [0, 0, 8, 8]]
)
    actual = solve_67385a82(input_grid)
    assert actual == expected


def test_67385a82_example_3():
    input_grid = ColoredGrid(values=
[[3, 3, 0, 0, 0, 0],
 [0, 3, 0, 0, 3, 0],
 [3, 0, 0, 0, 0, 0],
 [0, 3, 3, 0, 0, 0],
 [0, 3, 3, 0, 0, 3]]
    )
    expected = ColoredGrid(values=
[[8, 8, 0, 0, 0, 0],
 [0, 8, 0, 0, 3, 0],
 [3, 0, 0, 0, 0, 0],
 [0, 8, 8, 0, 0, 0],
 [0, 8, 8, 0, 0, 3]]
)
    actual = solve_67385a82(input_grid)
    assert actual == expected



def test_67385a82_test_case_0():
    input_grid = ColoredGrid(values=
[[3, 0, 3, 0, 3],
 [3, 3, 3, 0, 0],
 [0, 0, 0, 0, 3],
 [0, 3, 3, 0, 0],
 [0, 3, 3, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 0, 8, 0, 3],
 [8, 8, 8, 0, 0],
 [0, 0, 0, 0, 3],
 [0, 8, 8, 0, 0],
 [0, 8, 8, 0, 0]]
    )
    actual = solve_67385a82(input_grid)
    assert actual == expected

