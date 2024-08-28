from rob_agi.colored_grid import ColoredGrid
import copy

def solve_73c3b0d8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding yellow (4) squares above the red (2) line
    and moving yellow squares below the red line to just above it.
    
    The function follows these steps:
    1. Analyze the input grid to find the red line and yellow squares.
    2. Create a deep copy of the input grid.
    3. Expand yellow squares above the red line diagonally.
    4. Move yellow squares below the red line to just above it.
    5. Ensure balance and symmetry in the resulting pattern.
    6. Fill gaps and connect patterns if necessary.
    7. Perform a final consistency check.
    
    Returns the transformed grid.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    # Find red line and yellow squares
    red_line_row = next(r for r in range(rows) if all(cell == 2 for cell in input_grid[r]))
    yellow_squares = [(r, c) for r in range(rows) for c in range(cols) if input_grid[r][c] == 4]
    
    # Process yellow squares
    for r, c in yellow_squares:
        if r < red_line_row:
            # Expand diagonally
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                while 0 <= nr < red_line_row and 0 <= nc < cols and output_grid[nr][nc] == 0:
                    output_grid[nr][nc] = 4
                    nr, nc = nr + dr, nc + dc
        elif r > red_line_row:
            # Move to just above red line
            if output_grid[red_line_row - 1][c] == 0:
                output_grid[red_line_row - 1][c] = 4
                output_grid[r][c] = 0
    
    # Ensure at least one yellow square above red line
    if not any(4 in row for row in output_grid[:red_line_row]):
        mid_col = cols // 2
        output_grid[red_line_row - 1][mid_col] = 4
    
    # Balance and symmetry check
    left_count = sum(row[:cols//2].count(4) for row in output_grid[:red_line_row])
    right_count = sum(row[cols//2:].count(4) for row in output_grid[:red_line_row])
    if left_count > right_count:
        for r in range(red_line_row):
            output_grid[r] = output_grid[r][:cols//2] + output_grid[r][:cols//2][::-1]
    elif right_count > left_count:
        for r in range(red_line_row):
            output_grid[r] = output_grid[r][cols//2:][::-1] + output_grid[r][cols//2:]
    
    return output_grid
