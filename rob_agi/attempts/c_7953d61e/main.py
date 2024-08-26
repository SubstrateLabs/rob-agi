from rob_agi.colored_grid import ColoredGrid

def horizontal_flip(grid):
    return [row[::-1] for row in grid]

def vertical_flip(grid):
    return grid[::-1]

def both_flip(grid):
    return [row[::-1] for row in grid[::-1]]

def solve_7953d61e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 4x4 input grid into an 8x8 output grid by applying the following pattern:
    1. Copy the input to the top-left quadrant
    2. Apply horizontal flip to the top-right quadrant
    3. Apply vertical flip to the bottom-left quadrant
    4. Apply 180-degree rotation (both horizontal and vertical flip) to the bottom-right quadrant
    """
    input_values = input_grid.values
    new_grid = [[0 for _ in range(8)] for _ in range(8)]

    # Top-left quadrant (original)
    for i in range(4):
        for j in range(4):
            new_grid[i][j] = input_values[i][j]

    # Top-right quadrant (horizontal flip)
    flipped = horizontal_flip(input_values)
    for i in range(4):
        for j in range(4):
            new_grid[i][j+4] = flipped[i][j]

    # Bottom-left quadrant (vertical flip)
    flipped = vertical_flip(input_values)
    for i in range(4):
        for j in range(4):
            new_grid[i+4][j] = flipped[i][j]

    # Bottom-right quadrant (180-degree rotation)
    rotated = rotate_180(input_values)
    for i in range(4):
        for j in range(4):
            new_grid[i+4][j+4] = rotated[i][j]

    return ColoredGrid(values=new_grid)

def rotate_180(grid):
    return [row[::-1] for row in grid[::-1]]
