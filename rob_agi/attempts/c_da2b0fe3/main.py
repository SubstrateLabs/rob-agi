from rob_agi.colored_grid import ColoredGrid

def solve_da2b0fe3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by adding a horizontal green line to the input grid.
    The line is placed to intersect with the existing shape in the grid at its vertical center.
    
    1. Create a deep copy of the input grid.
    2. Find the bounding box of the non-zero elements.
    3. Calculate the vertical center of the shape.
    4. Add a horizontal green line across the entire grid at the calculated position.
    5. Return the modified grid with the added green line.
    """
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()
    
    # Find the bounding box of non-zero elements
    min_row, max_row = rows, -1
    for r in range(rows):
        for c in range(cols):
            if new_grid.values[r][c] != 0:
                min_row = min(min_row, r)
                max_row = max(max_row, r)
    
    # Calculate the vertical center of the shape
    middle_row = (min_row + max_row) // 2
    
    # Add horizontal green line
    for c in range(cols):
        new_grid.values[middle_row][c] = 3
    
    return new_grid
