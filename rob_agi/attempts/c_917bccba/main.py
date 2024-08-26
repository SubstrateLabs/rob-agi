from rob_agi.colored_grid import ColoredGrid

def solve_917bccba(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by rearranging a cross and rectangle pattern:
    1. Identifies the cross and rectangle colors and their positions
    2. Copies the rectangle to its original position
    3. Moves the vertical line of the cross to the right edge of the rectangle
    4. Adds horizontal lines of the cross color at the top and bottom of the rectangle,
       extending one cell beyond on both sides
    5. Preserves any parts of the cross above or below the rectangle
    6. Removes any internal vertical lines of the cross within the rectangle
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Analyze the input grid
    cross_color = None
    rect_color = None
    rect_left = cols
    rect_right = 0
    rect_top = rows
    rect_bottom = 0
    
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                if cross_color is None or input_grid.values[r][c] == cross_color:
                    cross_color = input_grid.values[r][c]
                else:
                    rect_color = input_grid.values[r][c]
                    rect_left = min(rect_left, c)
                    rect_right = max(rect_right, c)
                    rect_top = min(rect_top, r)
                    rect_bottom = max(rect_bottom, r)
    
    # Copy the rectangle
    for r in range(rect_top, rect_bottom + 1):
        for c in range(rect_left, rect_right + 1):
            if input_grid.values[r][c] == rect_color:
                new_grid.values[r][c] = rect_color
    
    # Draw the transformed cross
    # Vertical line at the right edge of the rectangle
    for r in range(rows):
        new_grid.values[r][rect_right] = cross_color
    
    # Horizontal lines at the top and bottom of the rectangle
    for c in range(rect_left - 1, rect_right + 2):
        if 0 <= c < cols:
            new_grid.values[rect_top][c] = cross_color
            new_grid.values[rect_bottom][c] = cross_color
    
    # Preserve parts of the cross above and below the rectangle
    for r in range(rows):
        if r < rect_top or r > rect_bottom:
            for c in range(cols):
                if input_grid.values[r][c] == cross_color:
                    new_grid.values[r][c] = cross_color
    
    return new_grid
