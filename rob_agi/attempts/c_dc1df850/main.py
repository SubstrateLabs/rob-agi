from rob_agi.colored_grid import ColoredGrid

def solve_dc1df850(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a 3x3 pattern of '1's around each '2',
    while preserving other non-zero values.

    The function:
    1. Creates a deep copy of the input grid.
    2. Iterates through each cell in the grid.
    3. If a cell contains '2', generates a 3x3 pattern of '1's around it.
    4. Stores any original non-zero values (except '2's) that are overwritten by the pattern.
    5. After processing all '2's, restores the stored original non-zero values.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    # List to store original non-zero values (except '2')
    original_values = []
    
    # Single pass: apply patterns and store original values
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 2:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            cell_value = grid.get_cell(nr, nc)
                            if cell_value != 0 and cell_value != 2:
                                original_values.append((nr, nc, cell_value))
                            if cell_value == 0:  # Only set to 1 if the cell is empty
                                grid.set_cell(nr, nc, 1)
    
    # Restore original non-zero values (except '2')
    for r, c, val in original_values:
        grid.set_cell(r, c, val)
    
    return grid
