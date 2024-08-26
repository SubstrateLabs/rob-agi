from rob_agi.colored_grid import ColoredGrid

def solve_d304284e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the d304284e challenge by identifying the original pattern in the input grid,
    then replicating it across the grid. The replication alternates between the original color
    and magenta (6), with black (0) cells separating each repetition both horizontally and vertically.
    The pattern starts from the top-left corner of the grid, and the original input is preserved.
    Partial patterns are added at the right and bottom edges if there's remaining space.
    """
    pattern, original_color = find_pattern(input_grid)
    if not pattern:
        return input_grid

    new_grid = create_replicated_grid(input_grid, pattern, original_color)
    preserve_original_input(input_grid, new_grid)

    return ColoredGrid(values=new_grid)

def find_pattern(grid: ColoredGrid):
    """Find the first non-zero pattern in the grid."""
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] != 0:
                color = grid.values[r][c]
                pattern = extract_pattern(grid, r, c, color)
                return pattern, color
    return None, 0

def extract_pattern(grid, start_row, start_col, color):
    pattern = []
    r, c = start_row, start_col
    while r < grid.num_rows and grid.values[r][start_col] == color:
        row = []
        while c < grid.num_cols and grid.values[r][c] == color:
            row.append(color)
            c += 1
        pattern.append(row)
        r += 1
        c = start_col
    return pattern

def create_replicated_grid(grid: ColoredGrid, pattern, original_color):
    new_grid = [[0 for _ in range(grid.num_cols)] for _ in range(grid.num_rows)]
    pattern_height, pattern_width = len(pattern), len(pattern[0])

    for r in range(0, grid.num_rows, pattern_height + 1):
        for c in range(0, grid.num_cols, pattern_width + 1):
            color = original_color if ((r // (pattern_height + 1)) + (c // (pattern_width + 1))) % 2 == 0 else 6
            stamp_pattern(new_grid, pattern, r, c, color)

    return new_grid

def stamp_pattern(grid, pattern, start_row, start_col, color):
    for r in range(len(pattern)):
        if start_row + r < len(grid):
            for c in range(len(pattern[0])):
                if start_col + c < len(grid[0]):
                    grid[start_row + r][start_col + c] = color

def preserve_original_input(input_grid, new_grid):
    for r in range(input_grid.num_rows):
        for c in range(input_grid.num_cols):
            if input_grid.values[r][c] != 0:
                new_grid[r][c] = input_grid.values[r][c]
