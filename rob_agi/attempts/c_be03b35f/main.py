from rob_agi.colored_grid import ColoredGrid

def solve_be03b35f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 5x5 input grid into a 2x2 output grid based on the distribution of blue cells.
    
    The function finds the bounding box of blue cells in the input grid and creates a 2x2 output grid
    where each cell represents an edge of the bounding box. A cell in the output grid is set to blue (1)
    if there's at least one blue cell along the corresponding edge of the bounding box in the input grid.
    
    Args:
    input_grid (ColoredGrid): A 5x5 input grid where 1 represents blue cells.
    
    Returns:
    ColoredGrid: A 2x2 output grid representing the edges of the blue cells' bounding box.
    """
    rows, cols = input_grid.get_dimensions()
    min_row, max_row = rows, 0
    min_col, max_col = cols, 0

    # Find the bounding box of blue cells
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) == 1:  # Blue cell
                min_row = min(min_row, r)
                max_row = max(max_row, r)
                min_col = min(min_col, c)
                max_col = max(max_col, c)

    # Create a 2x2 output grid
    output = [[0, 0], [0, 0]]

    # Check top edge
    if any(input_grid.get_cell(min_row, c) == 1 for c in range(min_col, max_col + 1)):
        output[0][0] = output[0][1] = 1

    # Check bottom edge
    if any(input_grid.get_cell(max_row, c) == 1 for c in range(min_col, max_col + 1)):
        output[1][0] = output[1][1] = 1

    # Check left edge
    if any(input_grid.get_cell(r, min_col) == 1 for r in range(min_row, max_row + 1)):
        output[0][0] = output[1][0] = 1

    # Check right edge
    if any(input_grid.get_cell(r, max_col) == 1 for r in range(min_row, max_row + 1)):
        output[0][1] = output[1][1] = 1

    return ColoredGrid(values=output)
