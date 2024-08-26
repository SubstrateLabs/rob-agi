from rob_agi.colored_grid import ColoredGrid

def solve_917bccba(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by rearranging a cross and shape pattern:
    1. Identifies the cross color and shape color
    2. Fills in the shape completely
    3. Moves the vertical line of the cross to the right edge of the shape
    4. Adds horizontal lines of the cross color at the top and bottom of the shape,
       extending to the edges of the grid
    5. Preserves any parts of the cross outside the shape
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
    
    # Fill the shape
    for r in range(shape_top, shape_bottom + 1):
        for c in range(shape_left, shape_right + 1):
            new_grid.values[r][c] = shape_color
    
    # Draw the transformed cross
    # Vertical line at the right edge of the shape
    for r in range(rows):
        new_grid.values[r][shape_right] = cross_color
    
    # Horizontal lines at the top and bottom of the shape
    for c in range(cols):
        if c < shape_left or c > shape_right:
            new_grid.values[shape_top][c] = cross_color
            new_grid.values[shape_bottom][c] = cross_color
    
    # Preserve original cross parts outside the shape
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == cross_color:
                if r < shape_top or r > shape_bottom or c < shape_left or c > shape_right:
                    new_grid.values[r][c] = cross_color
    
    return new_grid
