from rob_agi.colored_grid import ColoredGrid

def solve_da2b0fe3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by adding a green line to the input grid.
    The line is placed to intersect with the existing shape in the grid based on symmetry.
    
    1. Create a deep copy of the input grid.
    2. Find the bounding box of the non-zero elements.
    3. Calculate the center of the shape.
    4. Determine the shape's symmetry (vertical or horizontal).
    5. Add a vertical or horizontal green line across the entire grid through the center.
    6. Return the modified grid with the added green line.
    """
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()
    
    # Find the bounding box of non-zero elements
    min_row, max_row, min_col, max_col = rows, -1, cols, -1
    for r in range(rows):
        for c in range(cols):
            if new_grid.values[r][c] != 0:
                min_row = min(min_row, r)
                max_row = max(max_row, r)
                min_col = min(min_col, c)
                max_col = max(max_col, c)
    
    # Calculate the center of the shape
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2
    
    # Determine symmetry
    vertical_symmetry = 0
    horizontal_symmetry = 0
    
    for r in range(min_row, max_row + 1):
        for c in range(min_col, center_col):
            if new_grid.values[r][c] == new_grid.values[r][2 * center_col - c]:
                vertical_symmetry += 1
    
    for c in range(min_col, max_col + 1):
        for r in range(min_row, center_row):
            if new_grid.values[r][c] == new_grid.values[2 * center_row - r][c]:
                horizontal_symmetry += 1
    
    # Decide line orientation
    if vertical_symmetry >= horizontal_symmetry:
        # Add vertical green line
        for r in range(rows):
            new_grid.values[r][center_col] = 3
    else:
        # Add horizontal green line
        for c in range(cols):
            new_grid.values[center_row][c] = 3
    
    return new_grid
