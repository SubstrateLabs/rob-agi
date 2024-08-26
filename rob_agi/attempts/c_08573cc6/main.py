from rob_agi.colored_grid import ColoredGrid

def solve_08573cc6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a pattern with nested rectangles.
    
    The solution follows these steps:
    1. Analyze the input grid to determine dimensions, colors, and single colored square position.
    2. Calculate the pattern dimensions and position based on grid size.
    3. Create nested rectangles using the main color for the outer frame and alternating colors for inner frames.
    4. Incorporate the single colored square, preserving its relative position.
    5. Fill the inner spaces with the main color.
    6. Ensure some empty space around the edges of the grid.
    
    This pattern adapts to different grid sizes and single square positions while maintaining a consistent structure.
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
    start_row, start_col = 2, 2
    end_row, end_col = rows - 3, cols - 3
    
    # Draw outer rectangle
    for r in range(start_row, end_row + 1):
        new_grid.values[r][start_col] = main_color
        new_grid.values[r][end_col] = side_color
    for c in range(start_col, end_col + 1):
        new_grid.values[start_row][c] = main_color
        new_grid.values[end_row][c] = main_color
    
    # Draw inner rectangles
    num_inner_rectangles = min(2, (end_row - start_row - 2) // 2)
    for offset in range(1, num_inner_rectangles + 1):
        top = start_row + offset
        bottom = end_row - offset
        left = start_col + offset
        right = end_col - offset
        
        for r in range(top, bottom + 1):
            new_grid.values[r][left] = side_color if offset % 2 == 0 else main_color
            new_grid.values[r][right] = side_color if offset % 2 == 0 else main_color
        for c in range(left, right + 1):
            new_grid.values[top][c] = main_color
            new_grid.values[bottom][c] = main_color
    
    # Fill inner space
    for r in range(start_row + 1, end_row):
        for c in range(start_col + 1, end_col):
            if new_grid.values[r][c] == 0:
                new_grid.values[r][c] = main_color
    
    # Incorporate the single colored square
    if single_square:
        sr, sc, color = single_square
        relative_r = start_row + 1 + (sr - start_row) * (end_row - start_row - 2) // (rows - 4)
        relative_c = start_col + 1 + (sc - start_col) * (end_col - start_col - 2) // (cols - 4)
        new_grid.values[relative_r][relative_c] = color
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = relative_r + dr, relative_c + dc
            if start_row < nr < end_row and start_col < nc < end_col:
                new_grid.values[nr][nc] = main_color
    
    return new_grid
