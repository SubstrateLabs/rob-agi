import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_a79310a0.main import solve_a79310a0


def test_a79310a0_example_0():
    input_grid = ColoredGrid(values=
[[8, 8, 0, 0, 0],
 [8, 8, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0],
 [2, 2, 0, 0, 0],
 [2, 2, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0]]
)
    actual = solve_a79310a0(input_grid)
    assert actual == expected


def test_a79310a0_example_1():
    input_grid = ColoredGrid(values=
[[0, 8, 0], [0, 0, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0], [0, 2, 0], [0, 0, 0]]
)
    actual = solve_a79310a0(input_grid)
    assert actual == expected


def test_a79310a0_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0, 0, 0],
 [0, 8, 8, 8, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 2, 2, 2, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0]]
)
    actual = solve_a79310a0(input_grid)
    assert actual == expected



def test_a79310a0_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 8, 0, 0],
 [0, 8, 8, 0, 0],
 [0, 0, 8, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0],
 [0, 0, 2, 0, 0],
 [0, 2, 2, 0, 0],
 [0, 0, 2, 0, 0],
 [0, 0, 0, 0, 0]]
    )
    actual = solve_a79310a0(input_grid)
    assert actual == expected

