from rob_agi.colored_grid import ColoredGrid

def solve_be03b35f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 5x5 input grid into a 2x2 output grid based on the presence of blue cells in specific regions.
    
    The function checks for blue cells (1) in the following regions:
    - Left column (columns 0-1): If found, sets both left cells in the output to blue.
    - Right column (columns 3-4): If found, sets both right cells in the output to blue.
    - Top row (row 0): If found in left half, sets top-left cell to blue; if found in right half, sets top-right cell to blue.
    - Bottom row (row 4): If found in left half, sets bottom-left cell to blue; if found in right half, sets bottom-right cell to blue.
    
    Args:
    input_grid (ColoredGrid): A 5x5 input grid where 1 represents blue cells.
    
    Returns:
    ColoredGrid: A 2x2 output grid representing the presence of blue cells in each region.
    """
    output = [[0, 0], [0, 0]]

    # Check left column (columns 0-1)
    for r in range(5):
        if input_grid.get_cell(r, 0) == 1 or input_grid.get_cell(r, 1) == 1:
            output[0][0] = 1
            output[1][0] = 1
            break

    # Check right column (columns 3-4)
    for r in range(5):
        if input_grid.get_cell(r, 3) == 1 or input_grid.get_cell(r, 4) == 1:
            output[0][1] = 1
            output[1][1] = 1
            break

    # Check top row (row 0)
    for c in range(5):
        if c < 2 and input_grid.get_cell(0, c) == 1:
            output[0][0] = 1
        elif c > 2 and input_grid.get_cell(0, c) == 1:
            output[0][1] = 1

    # Check bottom row (row 4)
    for c in range(5):
        if c < 2 and input_grid.get_cell(4, c) == 1:
            output[1][0] = 1
        elif c > 2 and input_grid.get_cell(4, c) == 1:
            output[1][1] = 1

    return ColoredGrid(values=output)
