import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_68b16354.main import solve_68b16354


def test_68b16354_example_0():
    input_grid = ColoredGrid(values=
[[8, 1, 2, 1, 4],
 [4, 4, 2, 4, 8],
 [3, 7, 2, 4, 8],
 [2, 7, 7, 8, 7],
 [8, 7, 7, 4, 8]]
    )
    expected = ColoredGrid(values=
[[8, 7, 7, 4, 8],
 [2, 7, 7, 8, 7],
 [3, 7, 2, 4, 8],
 [4, 4, 2, 4, 8],
 [8, 1, 2, 1, 4]]
)
    actual = solve_68b16354(input_grid)
    assert actual == expected


def test_68b16354_example_1():
    input_grid = ColoredGrid(values=
[[7, 3, 3, 1, 2],
 [1, 8, 2, 4, 1],
 [2, 7, 8, 7, 2],
 [7, 7, 4, 1, 8],
 [8, 1, 7, 7, 1]]
    )
    expected = ColoredGrid(values=
[[8, 1, 7, 7, 1],
 [7, 7, 4, 1, 8],
 [2, 7, 8, 7, 2],
 [1, 8, 2, 4, 1],
 [7, 3, 3, 1, 2]]
)
    actual = solve_68b16354(input_grid)
    assert actual == expected


def test_68b16354_example_2():
    input_grid = ColoredGrid(values=
[[2, 7, 4, 3, 4, 8, 3],
 [2, 3, 7, 1, 2, 3, 3],
 [8, 7, 4, 3, 2, 2, 4],
 [1, 1, 2, 1, 4, 4, 7],
 [2, 4, 3, 1, 1, 4, 1],
 [4, 8, 7, 4, 4, 8, 2],
 [7, 3, 8, 4, 3, 2, 8]]
    )
    expected = ColoredGrid(values=
[[7, 3, 8, 4, 3, 2, 8],
 [4, 8, 7, 4, 4, 8, 2],
 [2, 4, 3, 1, 1, 4, 1],
 [1, 1, 2, 1, 4, 4, 7],
 [8, 7, 4, 3, 2, 2, 4],
 [2, 3, 7, 1, 2, 3, 3],
 [2, 7, 4, 3, 4, 8, 3]]
)
    actual = solve_68b16354(input_grid)
    assert actual == expected



def test_68b16354_test_case_0():
    input_grid = ColoredGrid(values=
[[2, 8, 1, 3, 2, 4, 1],
 [4, 4, 1, 1, 4, 3, 4],
 [1, 1, 1, 1, 4, 7, 3],
 [1, 1, 2, 3, 8, 1, 3],
 [4, 1, 1, 1, 7, 8, 4],
 [3, 2, 8, 4, 1, 8, 4],
 [1, 4, 7, 1, 2, 3, 4]]
    )
    expected = ColoredGrid(values=
[[1, 4, 7, 1, 2, 3, 4],
 [3, 2, 8, 4, 1, 8, 4],
 [4, 1, 1, 1, 7, 8, 4],
 [1, 1, 2, 3, 8, 1, 3],
 [1, 1, 1, 1, 4, 7, 3],
 [4, 4, 1, 1, 4, 3, 4],
 [2, 8, 1, 3, 2, 4, 1]]
    )
    actual = solve_68b16354(input_grid)
    assert actual == expected

