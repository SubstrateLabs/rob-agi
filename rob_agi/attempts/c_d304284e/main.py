from rob_agi.colored_grid import ColoredGrid

def solve_d304284e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the d304284e challenge by identifying the original pattern in the input grid,
    then using it as a stamp to create a new grid. The stamping process alternates
    between the original color and magenta, offsetting each row slightly to the right.
    The original pattern is always preserved in its original position.
    """
    # Find the original pattern
    pattern, top, left = find_pattern(input_grid)
    if not pattern:
        return input_grid  # Return the input if no pattern is found

    # Create the new grid by stamping the pattern
    new_grid = stamp_pattern(input_grid, pattern, top, left)

    return ColoredGrid(values=new_grid)

def find_pattern(grid: ColoredGrid):
    """Find the first non-zero pattern in the grid."""
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] != 0:
                color = grid.values[r][c]
                top, left = r, c
                bottom, right = r, c
                # Find the boundaries of the pattern
                while bottom + 1 < grid.num_rows and grid.values[bottom + 1][c] == color:
                    bottom += 1
                while right + 1 < grid.num_cols and grid.values[r][right + 1] == color:
                    right += 1
                pattern = [row[left:right+1] for row in grid.values[top:bottom+1]]
                return pattern, top, left
    return None, 0, 0

def stamp_pattern(grid: ColoredGrid, pattern, top, left):
    """Stamp the pattern across the grid, alternating colors and offsetting rows."""
    new_grid = [[0 for _ in range(grid.num_cols)] for _ in range(grid.num_rows)]
    pattern_height, pattern_width = len(pattern), len(pattern[0])
    original_color = pattern[0][0]

    for r in range(grid.num_rows):
        color = original_color if (r // pattern_height) % 2 == 0 else 6
        offset = (r // pattern_height) % pattern_width
        c = 0
        while c < grid.num_cols:
            for pr in range(pattern_height):
                for pc in range(pattern_width):
                    if r + pr < grid.num_rows and c + pc < grid.num_cols:
                        if pattern[pr][pc] != 0:
                            new_grid[r + pr][c + pc] = color
            c += pattern_width
        
    # Preserve the original pattern
    for r in range(len(pattern)):
        for c in range(len(pattern[0])):
            new_grid[top + r][left + c] = pattern[r][c]

    return new_grid
