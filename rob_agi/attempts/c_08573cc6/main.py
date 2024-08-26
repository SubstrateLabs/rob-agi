from rob_agi.colored_grid import ColoredGrid

def solve_08573cc6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a pattern with a rectangular frame.
    
    The solution follows these steps:
    1. Analyze the input grid to determine dimensions, colors, and single colored square position.
    2. Create a rectangular frame using the main color for most edges and the secondary color for the right edge.
    3. Position the single colored square within the frame, maintaining its relative position.
    4. Fill the inner space of the frame with the main color, leaving some empty space around the edges.
    5. Ensure the pattern adapts to different grid sizes while maintaining a consistent structure.
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
    
    # Determine frame size and position
    start_row, start_col = 2, 1
    end_row, end_col = rows - 4, cols - 6
    
    # Draw rectangular frame
    for r in range(start_row, end_row + 1):
        new_grid.values[r][start_col] = main_color
        new_grid.values[r][end_col] = side_color
    for c in range(start_col, end_col + 1):
        new_grid.values[start_row][c] = main_color
        new_grid.values[end_row][c] = main_color
    
    # Fill inner space
    for r in range(start_row + 1, end_row):
        for c in range(start_col + 1, end_col):
            new_grid.values[r][c] = main_color
    
    # Incorporate the single colored square
    if single_square:
        sr, sc, color = single_square
        relative_r = start_row + 1 + (sr - 2) * (end_row - start_row - 1) // (rows - 4)
        relative_c = start_col + 1 + (sc - 2) * (end_col - start_col - 1) // (cols - 4)
        new_grid.values[relative_r][relative_c] = color
        
        # Clear immediate surroundings
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = relative_r + dr, relative_c + dc
                if start_row < nr < end_row and start_col < nc < end_col and (dr != 0 or dc != 0):
                    new_grid.values[nr][nc] = 0
        
        # Add main color adjacent to single square
        new_grid.values[relative_r][relative_c - 1] = main_color
        new_grid.values[relative_r - 1][relative_c] = main_color
    
    return new_grid
