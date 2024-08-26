from rob_agi.colored_grid import ColoredGrid

def solve_45bbe264(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by drawing vertical and horizontal lines, and handling intersections.
    
    1. Draw vertical lines for all non-zero colors in the input grid.
    2. Draw horizontal lines for all non-zero colors in the input grid.
    3. When a horizontal line intersects with a different-colored vertical line,
       change the color to red (2) for this cell and all remaining cells in the row.
    
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
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                color = input_grid.values[r][c]
                for col in range(cols):
                    if output_grid.values[r][col] != 0 and output_grid.values[r][col] != color:
                        # We've hit an intersection, fill rest of row with red
                        for remaining_col in range(col, cols):
                            output_grid.values[r][remaining_col] = 2
                        break
                    else:
                        output_grid.values[r][col] = color
                break  # Move to next row after handling first non-zero

    return output_grid
