from rob_agi.colored_grid import ColoredGrid

def solve_d304284e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the d304284e challenge by identifying the original pattern in the input grid,
    then replicating it across the grid. The replication alternates between the original color
    and magenta (6), with black (0) rows separating each vertical repetition.
    The pattern starts from the top-left corner and the original input is preserved.
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
                width = get_pattern_width(grid, r, c, color)
                height = get_pattern_height(grid, r, c, color, width)
                pattern = [row[c:c+width] for row in grid.values[r:r+height]]
                return pattern, color
    return None, 0

def get_pattern_width(grid, row, col, color):
    width = 0
    while col + width < grid.num_cols and grid.values[row][col + width] == color:
        width += 1
    return width

def get_pattern_height(grid, row, col, color, width):
    height = 0
    while row + height < grid.num_rows and all(grid.values[row + height][col + i] == color for i in range(width)):
        height += 1
    return height

def create_replicated_grid(grid: ColoredGrid, pattern, original_color):
    new_grid = [[0 for _ in range(grid.num_cols)] for _ in range(grid.num_rows)]
    pattern_height, pattern_width = len(pattern), len(pattern[0])

    for r in range(0, grid.num_rows, pattern_height + 1):
        for c in range(0, grid.num_cols, pattern_width):
            color = original_color if (c // pattern_width) % 2 == 0 else 6
            stamp_pattern(new_grid, pattern, r, c, color)

    return new_grid

def stamp_pattern(grid, pattern, start_row, start_col, color):
    for r in range(len(pattern)):
        if start_row + r < len(grid):
            for c in range(len(pattern[0])):
                if start_col + c < len(grid[0]) and pattern[r][c] != 0:
                    grid[start_row + r][start_col + c] = color

def preserve_original_input(input_grid, new_grid):
    for r in range(input_grid.num_rows):
        for c in range(input_grid.num_cols):
            if input_grid.values[r][c] != 0:
                new_grid[r][c] = input_grid.values[r][c]
