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

def test_complex_shapes():
    input_grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    grid = ColoredGrid(values=input_grid)
    result = solve_776ffc46(grid)

    print("\nComplex Shapes Test")
    print("Input grid:")
    print_grid(input_grid)
    print("\nOutput grid:")
    print_grid(result.values)

def test_3x3_blue_square():
    input_grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 0, 3, 3, 3, 0, 0],
        [0, 2, 2, 2, 0, 3, 3, 3, 0, 0],
        [0, 2, 2, 2, 0, 3, 3, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    grid = ColoredGrid(values=input_grid)
    result = solve_776ffc46(grid)

    print("\n3x3 Blue Square Test")
    print("Input grid:")
    print_grid(input_grid)
    print("\nOutput grid:")
    print_grid(result.values)

test_vertical_line()
test_horizontal_line()
test_size_constraints()
test_gray_border()
test_color_prevalence()
test_complex_shapes()
test_3x3_blue_square()

def test_failing_case():
    input_grid = [
        [0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5],
        [0, 0, 2, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5],
        [0, 0, 2, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0, 2, 2, 2, 0, 5],
        [0, 0, 2, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0, 2, 2, 2, 0, 5],
        [0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5],
        [5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 2, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    grid = ColoredGrid(values=input_grid)
    result = solve_776ffc46(grid)

    print("\nFailing Test Case")
    print("Input grid:")
    print_grid(input_grid)
    print("\nOutput grid:")
    print_grid(result.values)

    # Add debug information
    blue_regions = []
    for i in range(len(input_grid)):
        for j in range(len(input_grid[0])):
            if input_grid[i][j] == 1 and not any(coord in region for region in blue_regions for coord in [(i,j)]):
                region = []
                queue = [(i,j)]
                while queue:
                    x, y = queue.pop(0)
                    if (x,y) not in region:
                        region.append((x,y))
                        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nx, ny = x+dx, y+dy
                            if 0 <= nx < len(input_grid) and 0 <= ny < len(input_grid[0]) and input_grid[nx][ny] == 1:
                                queue.append((nx,ny))
                blue_regions.append(region)
    
    print("\nBlue regions detected:")
    for i, region in enumerate(blue_regions):
        print(f"Region {i+1}: Size = {len(region)}, Coordinates = {region}")

test_failing_case()
