from rob_agi.colored_grid import ColoredGrid

def solve_be03b35f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 5x5 input grid into a 2x2 output grid based on the presence of blue cells in specific regions.
    
    The function checks for blue cells (1) in the following regions:
    - Top-left (rows 0-2, columns 0-2): If at least one blue cell is found, sets output[0][0] to blue.
    - Top-right (rows 0-2, columns 3-5): If at least one blue cell is found, sets output[0][1] to blue.
    - Bottom-left (rows 3-5, columns 0-2): If at least one blue cell is found, sets output[1][0] to blue.
    - Bottom-right (rows 3-5, columns 3-5): If at least one blue cell is found, sets output[1][1] to blue.
    
    The middle column and row are included in the checks for their respective quadrants.
    
    Args:
    input_grid (ColoredGrid): A 5x5 input grid where 1 represents blue cells.
    
    Returns:
    ColoredGrid: A 2x2 output grid representing the presence of blue cells in each region.
    """
    output = [[0, 0], [0, 0]]

    def check_region(start_row, end_row, start_col, end_col):
        return any(input_grid.get_cell(r, c) == 1 
                   for r in range(start_row, end_row) 
                   for c in range(start_col, end_col))

    # Check top-left region
    if check_region(0, 3, 0, 3):
        output[0][0] = 1

    # Check top-right region
    if check_region(0, 3, 2, 5):
        output[0][1] = 1

    # Check bottom-left region
    if check_region(2, 5, 0, 3):
        output[1][0] = 1

    # Check bottom-right region
    if check_region(2, 5, 2, 5):
        output[1][1] = 1

    return ColoredGrid(values=output)
