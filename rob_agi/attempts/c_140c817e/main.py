from rob_agi.colored_grid import ColoredGrid

def solve_140c817e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies all blue squares (value 1) in the input grid.
    2. For each blue square:
       a. Fills the entire row and column with 1s (blue).
       b. Sets diagonally adjacent cells to 3 (green), unless they are already part of another pattern.
       c. Changes the original blue square to 2 (red).
    3. Returns the transformed grid.
    """
    # Extract the grid data and create a deep copy
    grid = [row[:] for row in input_grid.values]
    rows, cols = len(grid), len(grid[0])
    
    # Identify all blue squares (value 1)
    blue_squares = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    
    for r, c in blue_squares:
        # Fill the entire row and column with 1s (blue)
        for i in range(cols):
            grid[r][i] = 1
        for i in range(rows):
            grid[i][c] = 1
        
        # Set diagonally adjacent cells to 3 (green)
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and input_grid.values[nr][nc] != 1:
                grid[nr][nc] = 3
        
        # Change the original blue square to 2 (red)
        grid[r][c] = 2
    
    # Return a new ColoredGrid with the transformed values
    return ColoredGrid(values=grid)
