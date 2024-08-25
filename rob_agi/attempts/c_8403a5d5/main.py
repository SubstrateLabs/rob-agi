from rob_agi.colored_grid import ColoredGrid

def solve_8403a5d5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following pattern:
    1. Identifies the anchor point (first non-zero value in the last row).
    2. Creates a new grid with the same dimensions, filled with zeros.
    3. Starting from the anchor column, fills every other column with the anchor value.
    4. Adds special pattern (5's) to the top and bottom rows:
       - Top row: First filled column after anchor, then every third filled column.
       - Bottom row: Third filled column after anchor, then every third filled column.
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
        top_start = anchor_col + 2
        bottom_start = anchor_col + 4
        for col in range(top_start, cols, 6):
            if col < cols:
                new_grid[0][col] = 5
        for col in range(bottom_start, cols, 6):
            if col < cols:
                new_grid[-1][col] = 5
        
        # Ensure the anchor value is preserved in the last row
        new_grid[-1][anchor_col] = anchor_value
    
    return ColoredGrid(values=new_grid)
