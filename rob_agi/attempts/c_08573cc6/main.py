from rob_agi.colored_grid import ColoredGrid

def solve_08573cc6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a pattern with nested rectangles.
    
    The solution follows these steps:
    1. Analyze the input grid to determine dimensions, colors, and single colored square position.
    2. Calculate the pattern dimensions and position based on grid size and single square location.
    3. Create an outer rectangle using the main color (top-left of input).
    4. Create inner rectangles, using the main color for top/bottom/left and side color for right.
    5. Incorporate the single colored square, adjusting nearby cells if needed.
    6. Fill the space between rectangles with the main color.
    7. Ensure some empty space around the edges of the grid.
    8. Adjust the pattern for visual balance and symmetry.
    
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
    
    # Draw rectangles
    for offset in range(0, (end_row - start_row) // 2 + 1):
        top = start_row + offset
        bottom = end_row - offset
        left = start_col + offset
        right = end_col - offset
        
        for r in range(top, bottom + 1):
            new_grid.values[r][left] = main_color
            new_grid.values[r][right] = side_color
        for c in range(left, right + 1):
            new_grid.values[top][c] = main_color
            new_grid.values[bottom][c] = main_color
    
    # Fill space between rectangles
    for r in range(start_row, end_row + 1):
        for c in range(start_col, end_col + 1):
            if new_grid.values[r][c] == 0:
                new_grid.values[r][c] = main_color
    
    # Incorporate the single colored square
    if single_square:
        sr, sc, color = single_square
        new_grid.values[sr][sc] = color
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = sr + dr, sc + dc
            if start_row <= nr <= end_row and start_col <= nc <= end_col:
                new_grid.values[nr][nc] = main_color
    
    # Adjust pattern for visual balance
    if cols > 7:
        for r in range(start_row, end_row + 1):
            new_grid.values[r][start_col - 1] = side_color
    
    return new_grid
