from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_776ffc46.main import solve_776ffc46

def print_grid(grid):
    for row in grid:
        print(' '.join(str(cell) for cell in row))

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
    print_grid(input_grid)
    print("\nOutput grid:")
    print_grid(result.values)

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
    print_grid(input_grid)
    print("\nOutput grid:")
    print_grid(result.values)

def test_size_constraints():
    input_grid = [
        [0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 2, 0, 3, 3, 3, 0, 0, 0],
        [2, 2, 2, 0, 3, 3, 3, 0, 0, 0],
        [2, 2, 2, 0, 3, 3, 3, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    grid = ColoredGrid(values=input_grid)
    result = solve_776ffc46(grid)

    print("\nSize Constraints Test")
    print("Input grid:")
    print_grid(input_grid)
    print("\nOutput grid:")
    print_grid(result.values)

def test_gray_border():
    input_grid = [
        [5, 5, 5, 5, 5],
        [5, 1, 1, 1, 5],
        [5, 1, 0, 1, 5],
        [5, 1, 1, 1, 5],
        [5, 5, 5, 5, 5]
    ]
    grid = ColoredGrid(values=input_grid)
    result = solve_776ffc46(grid)

    print("\nGray Border Test")
    print("Input grid:")
    print_grid(input_grid)
    print("\nOutput grid:")
    print_grid(result.values)

def test_color_prevalence():
    input_grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0, 3, 3, 3, 0, 0],
        [0, 2, 2, 0, 0, 3, 3, 3, 0, 0],
        [0, 0, 0, 0, 0, 3, 3, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    grid = ColoredGrid(values=input_grid)
    result = solve_776ffc46(grid)

    print("\nColor Prevalence Test")
    print("Input grid:")
    print_grid(input_grid)
    print("\nOutput grid:")
    print_grid(result.values)

test_vertical_line()
test_horizontal_line()
test_size_constraints()
test_gray_border()
test_color_prevalence()
