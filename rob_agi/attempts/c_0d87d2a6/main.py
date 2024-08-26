from rob_agi.colored_grid import ColoredGrid

def solve_0d87d2a6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by connecting blue dots with a minimal path and filling the left area.
    
    1. Finds all blue (1) dots in the grid.
    2. Creates vertical blue lines from each blue dot.
    3. Connects vertical lines horizontally at the level of each blue dot.
    4. Fills all cells to the left of the leftmost vertical blue line with blue.
    5. Converts red (2) blocks to blue if they intersect with the blue path.
    6. Preserves original blue dots and leaves unaffected cells unchanged.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Find blue dots
    blue_dots = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 1]
    if not blue_dots:
        return output_grid
    
    leftmost_col = min(c for _, c in blue_dots)
    
    # Step 2 & 3: Create vertical lines and connect them
    blue_dots.sort(key=lambda x: x[1])  # Sort by column
    for r in range(rows):
        for dot_row, dot_col in blue_dots:
            output_grid.values[r][dot_col] = 1  # Vertical line
        for i in range(len(blue_dots) - 1):
            if r == blue_dots[i][0]:  # If we're on a blue dot's row
                for c in range(blue_dots[i][1], blue_dots[i+1][1] + 1):
                    output_grid.values[r][c] = 1  # Horizontal connection
    
    # Step 4: Fill left area
    for r in range(rows):
        for c in range(leftmost_col):
            output_grid.values[r][c] = 1
    
    # Step 5: Handle red blocks
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 2 and output_grid.values[r][c] == 1:
                output_grid.values[r][c] = 1
            elif input_grid.values[r][c] == 2:
                output_grid.values[r][c] = 2
    
    return output_grid
