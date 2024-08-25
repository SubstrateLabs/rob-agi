import pytest
from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_8b28cd80.main import solve_8b28cd80


def test_8b28cd80_example_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0], [0, 4, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[4, 0, 4, 4, 4, 4, 4, 4, 4],
 [4, 0, 4, 0, 0, 0, 0, 0, 4],
 [4, 0, 4, 0, 4, 4, 4, 0, 4],
 [4, 0, 4, 0, 4, 0, 4, 0, 4],
 [4, 0, 4, 0, 4, 0, 4, 0, 4],
 [4, 0, 4, 0, 0, 0, 4, 0, 4],
 [4, 0, 4, 4, 4, 4, 4, 0, 4],
 [4, 0, 0, 0, 0, 0, 0, 0, 4],
 [4, 4, 4, 4, 4, 4, 4, 4, 4]]
)
    actual = solve_8b28cd80(input_grid)
    assert actual == expected


def test_8b28cd80_example_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0], [5, 0, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[5, 5, 5, 5, 5, 0, 5, 0, 5],
 [0, 0, 0, 0, 5, 0, 5, 0, 5],
 [5, 5, 5, 0, 5, 0, 5, 0, 5],
 [5, 0, 5, 0, 5, 0, 5, 0, 5],
 [5, 0, 5, 0, 5, 0, 5, 0, 5],
 [0, 0, 5, 0, 5, 0, 5, 0, 5],
 [5, 5, 5, 0, 5, 0, 5, 0, 5],
 [0, 0, 0, 0, 5, 0, 5, 0, 5],
 [5, 5, 5, 5, 5, 0, 5, 0, 5]]
)
    actual = solve_8b28cd80(input_grid)
    assert actual == expected


def test_8b28cd80_example_2():
    input_grid = ColoredGrid(values=
[[0, 3, 0], [0, 0, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[3, 0, 3, 0, 3, 0, 3, 0, 3],
 [3, 0, 3, 0, 0, 0, 3, 0, 3],
 [3, 0, 3, 3, 3, 3, 3, 0, 3],
 [3, 0, 0, 0, 0, 0, 0, 0, 3],
 [3, 3, 3, 3, 3, 3, 3, 3, 3],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [3, 3, 3, 3, 3, 3, 3, 3, 3],
 [0, 0, 0, 0, 0, 0, 0, 0, 0],
 [3, 3, 3, 3, 3, 3, 3, 3, 3]]
)
    actual = solve_8b28cd80(input_grid)
    assert actual == expected


def test_8b28cd80_example_3():
    input_grid = ColoredGrid(values=
[[0, 0, 0], [0, 0, 8], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[8, 0, 8, 0, 8, 0, 8, 8, 8],
 [8, 0, 8, 0, 8, 0, 8, 0, 0],
 [8, 0, 8, 0, 8, 0, 8, 0, 8],
 [8, 0, 8, 0, 8, 0, 8, 0, 8],
 [8, 0, 8, 0, 8, 0, 8, 0, 8],
 [8, 0, 8, 0, 8, 0, 8, 0, 0],
 [8, 0, 8, 0, 8, 0, 8, 8, 8],
 [8, 0, 8, 0, 8, 0, 0, 0, 0],
 [8, 0, 8, 0, 8, 8, 8, 8, 8]]
)
    actual = solve_8b28cd80(input_grid)
    assert actual == expected


def test_8b28cd80_example_4():
    input_grid = ColoredGrid(values=
[[0, 0, 7], [0, 0, 0], [0, 0, 0]]
    )
    expected = ColoredGrid(values=
[[7, 0, 7, 0, 7, 0, 7, 0, 7],
 [7, 0, 7, 0, 7, 0, 7, 0, 0],
 [7, 0, 7, 0, 7, 0, 7, 7, 7],
 [7, 0, 7, 0, 7, 0, 0, 0, 0],
 [7, 0, 7, 0, 7, 7, 7, 7, 7],
 [7, 0, 7, 0, 0, 0, 0, 0, 0],
 [7, 0, 7, 7, 7, 7, 7, 7, 7],
 [7, 0, 0, 0, 0, 0, 0, 0, 0],
 [7, 7, 7, 7, 7, 7, 7, 7, 7]]
)
    actual = solve_8b28cd80(input_grid)
    assert actual == expected



def test_8b28cd80_test_case_0():
    input_grid = ColoredGrid(values=
[[0, 0, 0], [0, 0, 0], [0, 0, 6]]
    )
    expected = ColoredGrid(values=
[[6, 0, 6, 6, 6, 6, 6, 6, 6],
 [6, 0, 6, 0, 0, 0, 0, 0, 0],
 [6, 0, 6, 0, 6, 6, 6, 6, 6],
 [6, 0, 6, 0, 6, 0, 0, 0, 0],
 [6, 0, 6, 0, 6, 0, 6, 6, 6],
 [6, 0, 6, 0, 6, 0, 6, 0, 0],
 [6, 0, 6, 0, 6, 0, 6, 0, 6],
 [6, 0, 6, 0, 6, 0, 6, 0, 6],
 [6, 0, 6, 0, 6, 0, 6, 0, 6]]
    )
    actual = solve_8b28cd80(input_grid)
    assert actual == expected


def test_8b28cd80_test_case_1():
    input_grid = ColoredGrid(values=
[[0, 0, 0], [0, 0, 0], [3, 0, 0]]
    )
    expected = ColoredGrid(values=
[[3, 3, 3, 3, 3, 3, 3, 3, 3],
 [0, 0, 0, 0, 0, 0, 0, 0, 3],
 [3, 3, 3, 3, 3, 3, 3, 0, 3],
 [0, 0, 0, 0, 0, 0, 3, 0, 3],
 [3, 3, 3, 3, 3, 0, 3, 0, 3],
 [0, 0, 0, 0, 3, 0, 3, 0, 3],
 [3, 3, 3, 0, 3, 0, 3, 0, 3],
 [3, 0, 3, 0, 3, 0, 3, 0, 3],
 [3, 0, 3, 0, 3, 0, 3, 0, 3]]
    )
    actual = solve_8b28cd80(input_grid)
    assert actual == expected

