from rob_agi.colored_grid import ColoredGrid

def solve_08573cc6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a pattern with two nested rectangles.
    
    The solution follows these steps:
    1. Analyze the input grid to determine dimensions and colors.
    2. Create an outer rectangle using the main color (top-left of input).
    3. Create an inner rectangle offset from the outer one.
    4. Use the side color (top-right of input) for the right sides of both rectangles.
    5. Add small extensions to the inner rectangle.
    6. Preserve the position of the single colored square from the input.
    
    This pattern adapts to different grid sizes while maintaining consistent proportions.
    """
    rows, cols = input_grid.get_dimensions()
    main_color = input_grid.values[0][0]
    side_color = input_grid.values[0][1]
    
    # Create a new grid filled with zeros (black)
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Draw outer rectangle
    for c in range(1, cols - 2):
        new_grid.values[2][c] = main_color  # Top
        new_grid.values[rows - 1][c] = main_color  # Bottom
    for r in range(2, rows):
        new_grid.values[r][1] = main_color  # Left side
        new_grid.values[r][cols - 2] = side_color  # Right side
    
    # Draw inner rectangle
    inner_top, inner_left = 4, 2
    inner_bottom, inner_right = rows - 2, cols - 3
    
    for c in range(inner_left, inner_right):
        new_grid.values[inner_top][c] = main_color  # Top
        new_grid.values[inner_bottom][c] = main_color  # Bottom
    for r in range(inner_top, inner_bottom + 1):
        new_grid.values[r][inner_left] = main_color  # Left side
        new_grid.values[r][inner_right] = side_color  # Right side
    
    # Add extensions
    new_grid.values[inner_top][inner_right] = main_color  # Top-right
    new_grid.values[inner_bottom][inner_left - 1] = main_color  # Bottom-left
    
    # Copy the single colored square from input
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] not in [0, main_color, side_color]:
                new_grid.values[r][c] = input_grid.values[r][c]
    
    return new_grid
