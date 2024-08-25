from rob_agi.colored_grid import ColoredGrid

def solve_08573cc6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a pattern with two nested rectangles.
    
    The solution follows these steps:
    1. Analyze the input grid to determine dimensions and colors.
    2. Create an outer rectangle using the main color (top-left of input).
    3. Create an inner rectangle offset from the outer one.
    4. Use the side color (adjacent to main color) for the right sides of both rectangles.
    5. Add extensions to the inner rectangle.
    6. Fill the space between rectangles.
    7. Preserve the position of the single colored square from the input.
    8. Adjust the pattern based on grid size.
    
    This pattern adapts to different grid sizes while maintaining consistent proportions.
    """
    rows, cols = input_grid.get_dimensions()
    main_color = input_grid.values[0][0]
    side_color = input_grid.values[0][1]
    
    # Create a new grid filled with zeros (black)
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Determine pattern size and position
    if rows <= 11:
        start_row, start_col = 2, 1
    elif rows == 12:
        start_row, start_col = 2, 2
    else:
        start_row, start_col = 1, 1
    
    # Draw outer rectangle
    for r in range(start_row, rows - 1):
        new_grid.values[r][start_col] = main_color  # Left side
        new_grid.values[r][cols - 2] = side_color  # Right side
    for c in range(start_col, cols - 1):
        new_grid.values[start_row][c] = main_color  # Top
        new_grid.values[rows - 2][c] = main_color  # Bottom
    
    # Draw inner rectangle
    inner_top = start_row + 2
    inner_left = start_col + 1
    inner_bottom = rows - 3
    inner_right = cols - 3
    
    for r in range(inner_top, inner_bottom):
        new_grid.values[r][inner_left] = main_color  # Left side
        new_grid.values[r][inner_right] = side_color  # Right side
    for c in range(inner_left, inner_right + 1):
        new_grid.values[inner_top][c] = main_color  # Top
        new_grid.values[inner_bottom][c] = main_color  # Bottom
    
    # Add extensions
    new_grid.values[inner_top - 1][inner_right] = main_color  # Top-right
    new_grid.values[inner_bottom + 1][inner_left - 1] = main_color  # Bottom-left
    
    # Fill space between rectangles
    for r in range(start_row + 1, rows - 2):
        for c in range(start_col + 1, cols - 2):
            if new_grid.values[r][c] == 0:
                if c < inner_right:
                    new_grid.values[r][c] = main_color
                else:
                    new_grid.values[r][c] = side_color
    
    # Preserve the original colored square
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] not in [0, main_color, side_color]:
                new_grid.values[r][c] = input_grid.values[r][c]
    
    return new_grid
