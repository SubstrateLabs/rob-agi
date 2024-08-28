from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_776ffc46.main import solve_776ffc46

def test_vertical_line():
    input_grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ]
    grid = ColoredGrid(values=input_grid)
    result = solve_776ffc46(grid)

    print("Vertical Line Test")
    print("Input grid:")
    for row in input_grid:
        print(row)

    print("\nOutput grid:")
    for row in result.values:
        print(row)

def test_horizontal_line():
    input_grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    grid = ColoredGrid(values=input_grid)
    result = solve_776ffc46(grid)

    print("\nHorizontal Line Test")
    print("Input grid:")
    for row in input_grid:
        print(row)

    print("\nOutput grid:")
    for row in result.values:
        print(row)

test_vertical_line()
test_horizontal_line()
