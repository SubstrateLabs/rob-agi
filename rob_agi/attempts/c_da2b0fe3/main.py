from rob_agi.colored_grid import ColoredGrid

def solve_da2b0fe3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by adding a green line to the input grid.
    The line is placed to intersect with the existing shape in the grid.
    If the shape is vertically oriented, add a horizontal green line.
    If the shape is horizontally oriented, add a vertical green line.
    
    1. Create a deep copy of the input grid.
    2. Determine the orientation of the existing shape.
    3. Add the green line in the appropriate direction.
    4. Return the modified grid with the added green line.
    """
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()
    
    # Determine the orientation of the existing shape
    vertical_count = sum(any(new_grid.values[r][c] != 0 for r in range(rows)) for c in range(cols))
    horizontal_count = sum(any(new_grid.values[r][c] != 0 for c in range(cols)) for r in range(rows))
    
    if vertical_count > horizontal_count:
        # Add horizontal green line
        middle_row = rows // 2
        for c in range(cols):
            new_grid.values[middle_row][c] = 3
    else:
        # Add vertical green line
        middle_col = cols // 2
        for r in range(rows):
            new_grid.values[r][middle_col] = 3
    
    return new_grid
