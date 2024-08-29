from rob_agi.colored_grid import ColoredGrid

def solve_8403a5d5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following pattern:
    1. Identifies the anchor point (first non-zero value in the last row).
    2. Creates a new grid with the same dimensions, filled with zeros.
    3. Starting from the anchor column, fills every other column with the anchor value.
    4. Adds special pattern (5's) to the top and bottom rows:
       - Top row: Second filled column after anchor, then every third filled column.
       - Bottom row: Fourth filled column after anchor, then every third filled column.
    5. Preserves all columns to the left of the anchor as zeros.
    6. Keeps the original anchor value in its position in the last row.
    """
    grid = input_grid.values
    rows, cols = len(grid), len(grid[0])
    
    # Find the anchor in the last row
    anchor_col = next((i for i, v in enumerate(grid[-1]) if v != 0), -1)
    anchor_value = grid[-1][anchor_col] if anchor_col != -1 else 0
    
    # Create a new grid filled with zeros
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    if anchor_col != -1:
        # Fill every other column starting from the anchor
        for col in range(anchor_col, cols, 2):
            for row in range(rows):
                new_grid[row][col] = anchor_value
        
        # Add special pattern (5's) to top and bottom rows
        filled_cols = [col for col in range(anchor_col, cols, 2)]
        if len(filled_cols) >= 2:
            new_grid[0][filled_cols[1]] = 5
        if len(filled_cols) >= 4:
            new_grid[-1][filled_cols[3]] = 5
        for i in range(4, len(filled_cols), 3):
            if i < len(filled_cols):
                new_grid[0][filled_cols[i]] = 5
            if i + 2 < len(filled_cols):
                new_grid[-1][filled_cols[i + 2]] = 5
    
    return ColoredGrid(values=new_grid)
