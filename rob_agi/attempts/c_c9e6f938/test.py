import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_c9e6f938.main import solve_c9e6f938


def test_c9e6f938_example_0():
    input_grid = ColoredGrid(values=
[[0, 7, 0], [0, 0, 7], [0, 7, 7]]
    )
    expected = ColoredGrid(values=
[[0, 7, 0, 0, 7, 0], [0, 0, 7, 7, 0, 0], [0, 7, 7, 7, 7, 0]]
)
    actual = solve_c9e6f938(input_grid)
    assert actual == expected


def test_c9e6f938_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0], [0, 7, 7], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0], [0, 7, 7, 7, 7, 0], [0, 0, 0, 0, 0, 0]]
)
    actual = solve_c9e6f938(input_grid)
    assert actual == expected


def test_c9e6f938_example_2():
    input_grid = ColoredGrid(values=
[[0, 0, 0], [7, 0, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0], [7, 0, 0, 0, 0, 7], [0, 0, 0, 0, 0, 0]]
)
    actual = solve_c9e6f938(input_grid)
    assert actual == expected



def test_c9e6f938_test_case_0():
    input_grid = ColoredGrid(values=
[[7, 7, 0], [0, 7, 0], [0, 0, 7]]
    )
    expected = ColoredGrid(values=
[[7, 7, 0, 0, 7, 7], [0, 7, 0, 0, 7, 0], [0, 0, 7, 7, 0, 0]]
    )
    actual = solve_c9e6f938(input_grid)
    assert actual == expected

