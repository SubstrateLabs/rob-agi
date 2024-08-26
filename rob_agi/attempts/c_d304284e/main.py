from rob_agi.colored_grid import ColoredGrid

def solve_d304284e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the d304284e challenge by identifying the original pattern in the input grid,
    then using it as a stamp to create a new grid. The stamping process alternates
    between the original color and magenta, with each row of stamps separated by a row of black.
    The original pattern is always preserved in its original position.
    """
    pattern, top, left = find_pattern(input_grid)
    if not pattern:
        return input_grid  # Return the input if no pattern is found

    new_grid = create_expanded_grid(input_grid, pattern, top, left)
    return ColoredGrid(values=new_grid)

def find_pattern(grid: ColoredGrid):
    """Find the first non-zero pattern in the grid."""
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] != 0:
                color = grid.values[r][c]
                top, left = r, c
                bottom, right = r, c
                while bottom + 1 < grid.num_rows and grid.values[bottom + 1][c] == color:
                    bottom += 1
                while right + 1 < grid.num_cols and grid.values[r][right + 1] == color:
                    right += 1
                pattern = [row[left:right+1] for row in grid.values[top:bottom+1]]
                return pattern, top, left
    return None, 0, 0

def create_expanded_grid(grid: ColoredGrid, pattern, top, left):
    """Create a new grid with the expanded pattern."""
    new_grid = [[0 for _ in range(grid.num_cols)] for _ in range(grid.num_rows)]
    pattern_height, pattern_width = len(pattern), len(pattern[0])
    original_color = pattern[0][0]

    row = 0
    while row < grid.num_rows:
        if row % (pattern_height + 1) == 0 and row != 0:
            row += 1
            continue

        color = original_color if ((row // (pattern_height + 1)) % 2 == 0) else 6
        col = 0
        while col < grid.num_cols:
            stamp_pattern(new_grid, pattern, row, col, color)
            col += pattern_width
        row += pattern_height

    # Preserve the original pattern
    for r in range(len(pattern)):
        for c in range(len(pattern[0])):
            new_grid[top + r][left + c] = pattern[r][c]

    return new_grid

def stamp_pattern(grid, pattern, start_row, start_col, color):
    """Stamp the pattern onto the grid at the specified position with the given color."""
    for r in range(len(pattern)):
        for c in range(len(pattern[0])):
            if start_row + r < len(grid) and start_col + c < len(grid[0]):
                if pattern[r][c] != 0:
                    grid[start_row + r][start_col + c] = color
