from rob_agi.colored_grid import ColoredGrid

def solve_9ddd00f0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by creating both horizontal and vertical symmetry for non-zero values.
    
    The function processes the grid as follows:
    1. Determines the center of the grid.
    2. Processes the grid in quadrants, using the bottom-right quadrant as reference.
    3. Copies non-zero values to corresponding positions in other quadrants, unless blocked by zeros.
    4. Preserves zeros from the input grid.
    5. Extends patterns where possible, respecting symmetry and zero-blocking rules.
    6. Handles odd-sized grids by preserving central rows/columns.
    
    This creates a pattern with both horizontal and vertical symmetry for non-zero values,
    while maintaining the original structure and preserving zero values as barriers.
    """
    height, width = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    center_row, center_col = height // 2, width // 2

    # Process the grid in quadrants
    for row in range(center_row, height):
        for col in range(center_col, width):
            value = input_grid.values[row][col]
            if value != 0:
                # Copy to all four quadrants if not blocked by zeros
                for r, c in [(row, col), (row, width-1-col), (height-1-row, col), (height-1-row, width-1-col)]:
                    if input_grid.values[r][c] == 0:
                        continue
                    output_grid.values[r][c] = value

    # Preserve zeros from input grid
    for row in range(height):
        for col in range(width):
            if input_grid.values[row][col] == 0:
                output_grid.values[row][col] = 0

    # Handle odd-sized grids
    if height % 2 != 0:
        output_grid.values[center_row] = input_grid.values[center_row][:]
    if width % 2 != 0:
        for row in range(height):
            output_grid.values[row][center_col] = input_grid.values[row][center_col]

    # Extend patterns
    for row in range(height):
        for col in range(width):
            if output_grid.values[row][col] == 0:
                # Check adjacent non-zero values and extend if possible
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < height and 0 <= nc < width and output_grid.values[nr][nc] != 0:
                        if input_grid.values[row][col] == 0:
                            continue
                        output_grid.values[row][col] = output_grid.values[nr][nc]
                        break

    return output_grid
