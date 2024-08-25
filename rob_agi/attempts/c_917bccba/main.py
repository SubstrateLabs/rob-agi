from rob_agi.colored_grid import ColoredGrid

def solve_917bccba(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by rearranging a cross and rectangle pattern:
    1. Identifies the cross and rectangle colors and their positions
    2. Shifts the rectangle one column to the right
    3. Extends the horizontal line of the cross across the width of the rectangle at its top
    4. Moves the vertical line of the cross to the right edge of the rectangle
    5. Preserves any parts of the cross above or below the rectangle
    6. Removes any internal vertical lines of the cross within the rectangle
    7. Extends the horizontal line of the cross one cell beyond the rectangle on both sides
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
    cross_vertical = None
    
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                if cross_color is None or input_grid.values[r][c] == cross_color:
                    cross_color = input_grid.values[r][c]
                    if cross_vertical is None or c == cross_vertical:
                        cross_vertical = c
                else:
                    rect_color = input_grid.values[r][c]
                    rect_left = min(rect_left, c)
                    rect_right = max(rect_right, c)
                    rect_top = min(rect_top, r)
                    rect_bottom = max(rect_bottom, r)
    
    # Shift the rectangle one column to the right
    for r in range(rect_top, rect_bottom + 1):
        for c in range(rect_right, rect_left - 1, -1):
            new_grid.values[r][c + 1] = input_grid.values[r][c]
    
    # Draw the transformed cross
    # Horizontal line at the top of the rectangle, extending one cell beyond on both sides
    for c in range(rect_left, rect_right + 3):
        new_grid.values[rect_top][c] = cross_color
    # Vertical line at the right edge of the shifted rectangle
    for r in range(rows):
        new_grid.values[r][rect_right + 1] = cross_color
    
    # Preserve parts of the cross above and below the rectangle
    for r in range(rows):
        if r < rect_top or r > rect_bottom:
            new_grid.values[r][cross_vertical] = input_grid.values[r][cross_vertical]
    
    # Remove any internal vertical lines of the cross within the rectangle
    for r in range(rect_top + 1, rect_bottom):
        for c in range(rect_left + 1, rect_right + 1):
            if new_grid.values[r][c] == cross_color and c != rect_right + 1:
                new_grid.values[r][c] = 0
    
    return new_grid
