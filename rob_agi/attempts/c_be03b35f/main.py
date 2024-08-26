from rob_agi.colored_grid import ColoredGrid

def solve_be03b35f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 5x5 input grid into a 2x2 output grid based on the presence of blue cells in specific quadrants.
    
    The function checks for blue cells (1) in the following quadrants:
    - Top-left (rows 0-1, columns 0-1): If found, sets output[0][0] to blue.
    - Top-right (rows 0-1, columns 3-4): If found, sets output[0][1] to blue.
    - Bottom-left (rows 3-4, columns 0-1): If found, sets output[1][0] to blue.
    - Bottom-right (rows 3-4, columns 3-4): If found, sets output[1][1] to blue.
    
    Args:
    input_grid (ColoredGrid): A 5x5 input grid where 1 represents blue cells.
    
    Returns:
    ColoredGrid: A 2x2 output grid representing the presence of blue cells in each quadrant.
    """
    output = [[0, 0], [0, 0]]

    # Check top-left quadrant
    for r in range(2):
        for c in range(2):
            if input_grid.get_cell(r, c) == 1:
                output[0][0] = 1
                break
        if output[0][0] == 1:
            break

    # Check top-right quadrant
    for r in range(2):
        for c in range(3, 5):
            if input_grid.get_cell(r, c) == 1:
                output[0][1] = 1
                break
        if output[0][1] == 1:
            break

    # Check bottom-left quadrant
    for r in range(3, 5):
        for c in range(2):
            if input_grid.get_cell(r, c) == 1:
                output[1][0] = 1
                break
        if output[1][0] == 1:
            break

    # Check bottom-right quadrant
    for r in range(3, 5):
        for c in range(3, 5):
            if input_grid.get_cell(r, c) == 1:
                output[1][1] = 1
                break
        if output[1][1] == 1:
            break

    return ColoredGrid(values=output)
