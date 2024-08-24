from rob_agi.colored_grid import ColoredGrid

def solve_917bccba(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by rearranging a cross and rectangle pattern:
    1. Shifts the rectangle one column to the left
    2. Extends the horizontal line of the cross across the full width at the top of the rectangle
    3. Moves the vertical line of the cross to the right edge of the rectangle
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
                    if cross_vertical is None or c < cross_vertical:
                        cross_vertical = c
                else:
                    rect_color = input_grid.values[r][c]
                    rect_left = min(rect_left, c)
                    rect_right = max(rect_right, c)
                    rect_top = min(rect_top, r)
                    rect_bottom = max(rect_bottom, r)
    
    # Draw the transformed rectangle
    for r in range(rect_top, rect_bottom + 1):
        for c in range(rect_left - 1, rect_right):
            new_grid.values[r][c] = input_grid.values[r][c + 1]
    
    # Draw the transformed cross
    for c in range(cols):
        new_grid.values[rect_top][c] = cross_color
    for r in range(rows):
        new_grid.values[r][rect_right - 1] = cross_color
    
    return new_grid
