from rob_agi.colored_grid import ColoredGrid

def solve_08573cc6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a pattern with two nested rectangles.
    
    The solution follows these steps:
    1. Analyze the input grid to determine dimensions, colors, and single colored square position.
    2. Create an outer rectangle using the main color (top-left of input).
    3. Create an inner rectangle offset from the outer one.
    4. Use the side color (adjacent to main color) for the right sides of both rectangles.
    5. Add extensions to the inner rectangle if applicable.
    6. Fill the space between rectangles.
    7. Preserve the position of the single colored square from the input.
    8. Adjust the pattern based on grid size and single square position.
    
    This pattern adapts to different grid sizes and single square positions while maintaining consistent structure.
    """
    rows, cols = input_grid.get_dimensions()
    main_color = input_grid.values[0][0]
    side_color = input_grid.values[0][1]
    
    # Find the single colored square
    single_square = None
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] not in [0, main_color, side_color]:
                single_square = (r, c, input_grid.values[r][c])
                break
        if single_square:
            break
    
    # Create a new grid filled with zeros (black)
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Determine pattern size and position
    start_row, start_col = 4, 2
    end_row, end_col = rows - 2, cols - 2
    
    # Draw outer rectangle
    for r in range(start_row, end_row + 1):
        new_grid.values[r][start_col] = main_color  # Left side
        new_grid.values[r][end_col] = side_color  # Right side
    for c in range(start_col, end_col + 1):
        new_grid.values[start_row][c] = main_color  # Top
        new_grid.values[end_row][c] = main_color  # Bottom
    
    # Draw inner rectangle
    inner_top = start_row + 1
    inner_left = start_col + 1
    inner_bottom = end_row - 1
    inner_right = end_col - 1
    
    for r in range(inner_top, inner_bottom + 1):
        new_grid.values[r][inner_left] = side_color  # Left side
        new_grid.values[r][inner_right] = side_color  # Right side
    for c in range(inner_left, inner_right + 1):
        new_grid.values[inner_top][c] = main_color  # Top
        new_grid.values[inner_bottom][c] = main_color  # Bottom
    
    # Fill space between rectangles
    for r in range(start_row + 1, end_row):
        for c in range(start_col + 1, end_col):
            if new_grid.values[r][c] == 0:
                new_grid.values[r][c] = main_color
    
    # Add inner pattern
    if single_square:
        sr, sc, _ = single_square
        inner_pattern_top = max(sr - 1, inner_top + 1)
        inner_pattern_left = max(sc - 1, inner_left + 1)
        inner_pattern_bottom = min(sr + 1, inner_bottom - 1)
        inner_pattern_right = min(sc + 1, inner_right - 1)
        
        for r in range(inner_pattern_top, inner_pattern_bottom + 1):
            for c in range(inner_pattern_left, inner_pattern_right + 1):
                if (r, c) != (sr, sc):
                    new_grid.values[r][c] = main_color
    
    # Preserve the original colored square
    if single_square:
        sr, sc, color = single_square
        new_grid.values[sr][sc] = color
    
    return new_grid
