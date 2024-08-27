from rob_agi.colored_grid import ColoredGrid

def solve_1d0a4b61(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid pattern challenge by identifying the repeating pattern
    and restoring corrupted (black) areas.

    1. Analyzes the input grid to find the repeating pattern.
    2. Creates a pattern template.
    3. Restores the grid by applying the pattern template to corrupted areas.
    4. Ensures the border is consistently filled.

    Args:
    input_grid (ColoredGrid): The input grid with corrupted areas.

    Returns:
    ColoredGrid: The restored grid with the pattern applied consistently.
    """
    rows, cols = input_grid.get_dimensions()
    border_color = input_grid.values[0][0]  # Assuming border color is consistent

    # Find the repeating pattern
    pattern = find_pattern(input_grid)
    pattern_rows, pattern_cols = len(pattern), len(pattern[0])

    # Create a new grid and restore it
    restored_grid = input_grid.deep_copy()
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 0 or (r == 0 or r == rows-1 or c == 0 or c == cols-1):
                # If it's a corrupted cell or on the border, fill with the pattern
                pattern_r, pattern_c = r % pattern_rows, c % pattern_cols
                restored_grid.values[r][c] = pattern[pattern_r][pattern_c]

    return restored_grid

def find_pattern(grid: ColoredGrid) -> list:
    """
    Finds the repeating pattern in the grid.
    """
    rows, cols = grid.get_dimensions()
    for pattern_size in range(2, min(rows, cols) // 2):
        pattern = [row[:pattern_size] for row in grid.values[:pattern_size]]
        if is_valid_pattern(grid, pattern):
            return pattern
    return grid.values  # If no pattern found, return the whole grid as pattern

def is_valid_pattern(grid: ColoredGrid, pattern: list) -> bool:
    """
    Checks if the given pattern is valid for the entire grid.
    """
    rows, cols = grid.get_dimensions()
    pattern_rows, pattern_cols = len(pattern), len(pattern[0])
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != 0:  # Skip corrupted cells
                pattern_r, pattern_c = r % pattern_rows, c % pattern_cols
                if grid.values[r][c] != pattern[pattern_r][pattern_c]:
                    return False
    return True
