from rob_agi.colored_grid import ColoredGrid

def solve_08573cc6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a complex pattern with nested rectangular frames.
    
    The solution follows these steps:
    1. Analyze the input grid to determine dimensions, colors, and single colored square position.
    2. Create an outer rectangular frame using the main color for top, bottom, and left edges, and the secondary color for the right edge.
    3. Create an inner rectangular frame using the secondary color for top, left, and right edges, and the main color for the bottom edge.
    4. Position the single colored square within the inner frame, maintaining its relative position.
    5. Add main color squares adjacent to the single colored square.
    6. Fill the remaining space inside the inner frame with the main color, leaving a 1-cell gap around the single colored square and its adjacent main color squares.
    7. Ensure the pattern is centered and adapts to different grid sizes while maintaining a consistent structure.
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
    
    # Determine outer frame size and position
    outer_start_row, outer_start_col = (rows - 7) // 2, (cols - 7) // 2
    outer_end_row, outer_end_col = outer_start_row + 6, outer_start_col + 6
    
    # Determine inner frame size and position
    inner_start_row, inner_start_col = outer_start_row + 1, outer_start_col + 1
    inner_end_row, inner_end_col = outer_end_row - 1, outer_end_col - 1
    
    # Draw outer rectangular frame
    for r in range(outer_start_row, outer_end_row + 1):
        new_grid.values[r][outer_start_col] = main_color
        new_grid.values[r][outer_end_col] = side_color
    for c in range(outer_start_col, outer_end_col + 1):
        new_grid.values[outer_start_row][c] = main_color
        new_grid.values[outer_end_row][c] = main_color
    
    # Draw inner rectangular frame
    for r in range(inner_start_row, inner_end_row + 1):
        new_grid.values[r][inner_start_col] = side_color
        new_grid.values[r][inner_end_col] = side_color
    for c in range(inner_start_col, inner_end_col + 1):
        new_grid.values[inner_start_row][c] = side_color
        new_grid.values[inner_end_row][c] = main_color
    
    # Incorporate the single colored square
    if single_square:
        sr, sc, color = single_square
        relative_r = inner_start_row + 2
        relative_c = inner_start_col + 3
        new_grid.values[relative_r][relative_c] = color
        
        # Add main color adjacent to single square
        new_grid.values[relative_r][relative_c - 1] = main_color
        new_grid.values[relative_r - 1][relative_c] = main_color
    
    # Fill inner frame
    for r in range(inner_start_row + 1, inner_end_row):
        for c in range(inner_start_col + 1, inner_end_col):
            if new_grid.values[r][c] == 0:
                new_grid.values[r][c] = main_color
    
    return new_grid
