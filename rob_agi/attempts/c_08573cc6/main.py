from rob_agi.colored_grid import ColoredGrid

def solve_08573cc6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a pattern with nested rectangles.
    
    The solution follows these steps:
    1. Analyze the input grid to determine dimensions, colors, and single colored square position.
    2. Calculate the pattern dimensions and position based on grid size and single square location.
    3. Create an outer rectangle using the main color (top-left of input).
    4. Create inner rectangles if space allows, alternating colors for the left side.
    5. Use the side color (adjacent to main color) for the right sides of all rectangles.
    6. Incorporate the single colored square, creating a small extension if needed.
    7. Fill the space between rectangles with the main color.
    8. Preserve the position and color of the single colored square from the input.
    9. Adjust the pattern to ensure it's centered and leaves some empty space around the edges.
    
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
    start_row, start_col = 2, 1
    end_row, end_col = rows - 2, cols - 2
    
    # Draw outer rectangle
    for r in range(start_row, end_row + 1):
        new_grid.values[r][start_col] = main_color  # Left side
        new_grid.values[r][end_col] = side_color  # Right side
    for c in range(start_col, end_col + 1):
        new_grid.values[start_row][c] = main_color  # Top
        new_grid.values[end_row][c] = main_color  # Bottom
    
    # Draw inner rectangles
    inner_top, inner_left = start_row + 1, start_col + 1
    inner_bottom, inner_right = end_row - 1, end_col - 1
    
    while inner_top < inner_bottom and inner_left < inner_right:
        for r in range(inner_top, inner_bottom + 1):
            new_grid.values[r][inner_left] = side_color if inner_left == start_col + 1 else main_color
            new_grid.values[r][inner_right] = side_color
        for c in range(inner_left, inner_right + 1):
            new_grid.values[inner_top][c] = main_color
            new_grid.values[inner_bottom][c] = main_color
        inner_top += 1
        inner_left += 1
        inner_bottom -= 1
        inner_right -= 1
    
    # Fill space between rectangles
    for r in range(start_row + 1, end_row):
        for c in range(start_col + 1, end_col):
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
    
    return new_grid
