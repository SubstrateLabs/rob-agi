from rob_agi.colored_grid import ColoredGrid

def solve_be03b35f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 5x5 input grid into a 2x2 output grid based on the presence of blue cells in specific regions.
    
    The function checks for blue cells (1) in the following regions:
    - Top-left (rows 0-1, columns 0-1): If found, sets output[0][0] to blue.
    - Top-right (rows 0-1, columns 3-4): If found, sets output[0][1] to blue.
    - Bottom-left (rows 3-4, columns 0-1): If found, sets output[1][0] to blue.
    - Bottom-right (rows 3-4, columns 3-4): If found, sets output[1][1] to blue.
    
    The middle column and row are ignored in the input grid.
    
    Args:
    input_grid (ColoredGrid): A 5x5 input grid where 1 represents blue cells.
    
    Returns:
    ColoredGrid: A 2x2 output grid representing the presence of blue cells in each region.
    """
    output = [[0, 0], [0, 0]]

    # Check top-left region
    if any(input_grid.get_cell(r, c) == 1 for r in range(2) for c in range(2)):
        output[0][0] = 1

    # Check top-right region
    if any(input_grid.get_cell(r, c) == 1 for r in range(2) for c in range(3, 5)):
        output[0][1] = 1

    # Check bottom-left region
    if any(input_grid.get_cell(r, c) == 1 for r in range(3, 5) for c in range(2)):
        output[1][0] = 1

    # Check bottom-right region
    if any(input_grid.get_cell(r, c) == 1 for r in range(3, 5) for c in range(3, 5)):
        output[1][1] = 1

    return ColoredGrid(values=output)
