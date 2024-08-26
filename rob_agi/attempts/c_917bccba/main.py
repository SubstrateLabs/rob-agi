from rob_agi.colored_grid import ColoredGrid

def solve_917bccba(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by rearranging a cross and shape pattern:
    1. Identifies the cross color and shape color
    2. Moves the shape to the left side of the grid
    3. Moves the vertical line of the cross to one column right of the shape's right edge
    4. Adds a horizontal line of the cross color at the top of the shape,
       extending to the edges of the grid
    5. Preserves the vertical parts of the cross outside the shape
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Analyze the input grid
    colors = set()
    shape_left = cols
    shape_right = 0
    shape_top = rows
    shape_bottom = 0
    
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                colors.add(input_grid.values[r][c])
                shape_left = min(shape_left, c)
                shape_right = max(shape_right, c)
                shape_top = min(shape_top, r)
                shape_bottom = max(shape_bottom, r)
    
    cross_color, shape_color = colors
    if input_grid.values[shape_top][shape_left] != shape_color:
        cross_color, shape_color = shape_color, cross_color
    
    shape_width = shape_right - shape_left + 1
    
    # Fill the shape (moved to the left)
    for r in range(shape_top, shape_bottom + 1):
        for c in range(shape_width):
            new_grid.values[r][c] = shape_color
    
    # Draw the transformed cross
    # Vertical line one column right of the shape's right edge
    vertical_line_col = shape_width
    for r in range(rows):
        new_grid.values[r][vertical_line_col] = cross_color
    
    # Horizontal line at the top of the shape
    for c in range(cols):
        new_grid.values[shape_top][c] = cross_color
    
    # Preserve original vertical cross parts outside the shape
    for r in range(rows):
        if r < shape_top or r > shape_bottom:
            new_grid.values[r][vertical_line_col] = cross_color
    
    return new_grid
