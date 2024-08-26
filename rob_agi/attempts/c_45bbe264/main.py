from rob_agi.colored_grid import ColoredGrid

def solve_45bbe264(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by drawing vertical and horizontal lines, and handling intersections.
    
    1. Draw vertical lines for all non-zero colors in the input grid.
    2. Draw horizontal lines for all non-zero colors in the input grid.
    3. When a horizontal line intersects with a different-colored vertical line,
       change the color to red (2) for this cell and stop drawing the horizontal line.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    # Step 1: Draw vertical lines
    for c in range(cols):
        for r in range(rows):
            if input_grid.values[r][c] != 0:
                color = input_grid.values[r][c]
                for row in range(rows):
                    output_grid.values[row][c] = color
                break  # Move to next column after finding first non-zero

    # Step 2 & 3: Draw horizontal lines and handle intersections
    for r in range(rows):
        left_color = 0
        left_col = -1
        right_col = -1
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                if left_color == 0:
                    left_color = input_grid.values[r][c]
                    left_col = c
                right_col = c
        
        if left_col != -1 and right_col != -1:
            for c in range(left_col, right_col + 1):
                if output_grid.values[r][c] == 0:
                    output_grid.values[r][c] = left_color
                elif output_grid.values[r][c] != left_color:
                    output_grid.values[r][c] = 2
                    break  # Stop drawing the horizontal line

    return output_grid
