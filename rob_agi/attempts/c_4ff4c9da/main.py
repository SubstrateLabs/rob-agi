from rob_agi.colored_grid import ColoredGrid

def solve_4ff4c9da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating four-way symmetry for all sky blue (8) cells.
    
    The function identifies all sky blue cells and mirrors them to all four quadrants
    of the grid, only replacing black (0) or blue (1) cells. Red (2) cells are preserved.
    
    Steps:
    1. Identify the center of the grid
    2. Scan the input grid for sky blue cells
    3. Mirror each sky blue cell to all four quadrants
    4. Create a new grid with the transformed pattern
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    rows, cols = input_grid.get_dimensions()
    center_row, center_col = rows // 2, cols // 2
    new_grid = input_grid.deep_copy()
    
    sky_blue_cells = []
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) == 8:
                sky_blue_cells.append((r, c))
    
    for row, col in sky_blue_cells:
        mirror_positions = [
            (row, col),
            (row, 2 * center_col - col),
            (2 * center_row - row, col),
            (2 * center_row - row, 2 * center_col - col)
        ]
        
        for r, c in mirror_positions:
            if 0 <= r < rows and 0 <= c < cols:
                if new_grid.get_cell(r, c) in [0, 1]:
                    new_grid.set_cell(r, c, 8)
    
    return new_grid
