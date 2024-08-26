from rob_agi.colored_grid import ColoredGrid

def solve_45bbe264(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by drawing lines and handling intersections.
    
    1. For each non-zero color in the input:
       - If in an even row, draw a horizontal line across the entire row.
       - If in an odd column, draw a vertical line down the entire column.
    2. At intersections of different colors, place a red (2) square.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    # First pass: Draw lines
    for r in range(rows):
        for c in range(cols):
            color = input_grid.values[r][c]
            if color != 0:
                if r % 2 == 0:  # Even row
                    for col in range(cols):
                        if output_grid.values[r][col] == 0:
                            output_grid.values[r][col] = color
                if c % 2 == 1:  # Odd column
                    for row in range(rows):
                        if output_grid.values[row][c] == 0:
                            output_grid.values[row][c] = color

    # Second pass: Handle intersections
    for r in range(rows):
        for c in range(cols):
            row_color = output_grid.values[r][c]
            if r > 0 and c > 0:
                col_color = output_grid.values[r-1][c]
                if row_color != col_color and row_color != 0 and col_color != 0:
                    output_grid.values[r][c] = 2  # Red for intersection

    return output_grid
