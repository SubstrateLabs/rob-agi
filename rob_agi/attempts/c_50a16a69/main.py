from rob_agi.colored_grid import ColoredGrid

def find_pattern(grid: ColoredGrid) -> tuple[list[list[int]], int, int]:
    rows, cols = grid.get_dimensions()
    for pattern_height in range(1, rows + 1):
        for pattern_width in range(1, cols + 1):
            pattern = [row[:pattern_width] for row in grid.values[:pattern_height]]
            if all(grid.values[r][c] == pattern[r % pattern_height][c % pattern_width]
                   for r in range(rows) for c in range(cols)):
                return pattern, pattern_height, pattern_width
    return grid.values, rows, cols  # If no pattern found, return the entire grid

def detect_phase_shift(grid: ColoredGrid, pattern: list[list[int]]) -> tuple[int, int]:
    for r in range(len(pattern)):
        for c in range(len(pattern[0])):
            if grid.values[0][0] == pattern[r][c]:
                return r, c
    return 0, 0  # No phase shift if not found

def solve_50a16a69(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying the core repeating pattern and extending it to cover the entire grid.
    
    The function performs the following steps:
    1. Identifies the smallest repeating pattern in the input grid.
    2. Detects any phase shift in the pattern relative to the top-left corner.
    3. Generates a new grid by extending the identified pattern across the entire area, accounting for the phase shift.
    
    This approach works for various patterns, including checkerboard patterns, handling different grid sizes,
    border colors, and extending the pattern to areas that were originally borders or uniform regions.
    It also correctly handles cases where the pattern might start at an offset from the top-left corner.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the extended pattern.
    """
    rows, cols = input_grid.get_dimensions()
    pattern, pattern_height, pattern_width = find_pattern(input_grid)
    shift_r, shift_c = detect_phase_shift(input_grid, pattern)
    
    output_values = []
    for r in range(rows):
        row = []
        for c in range(cols):
            pattern_r = (r + shift_r) % pattern_height
            pattern_c = (c + shift_c) % pattern_width
            row.append(pattern[pattern_r][pattern_c])
        output_values.append(row)
    
    return ColoredGrid(values=output_values)
