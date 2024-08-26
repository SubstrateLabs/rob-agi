from rob_agi.colored_grid import ColoredGrid

def solve_45bbe264(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by drawing vertical and horizontal lines, handling intersections,
    and preserving original non-zero values.
    
    1. Draw vertical lines for all non-zero colors in the input grid.
    2. Draw horizontal lines for all non-zero colors in the input grid.
    3. Change intersections of different colors to red (2).
    4. Preserve original non-zero values from the input grid.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    vertical_lines = set()
    horizontal_lines = set()

    # Step 1: Draw vertical lines
    for c in range(cols):
        for r in range(rows):
            if input_grid.values[r][c] != 0:
                color = input_grid.values[r][c]
                for row in range(rows):
                    output_grid.values[row][c] = color
                vertical_lines.add(c)
                break  # Move to next column after finding first non-zero

    # Step 2: Draw horizontal lines
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                color = input_grid.values[r][c]
                for col in range(cols):
                    if output_grid.values[r][col] == 0 or output_grid.values[r][col] == color:
                        output_grid.values[r][col] = color
                    else:
                        output_grid.values[r][col] = 2  # Intersection
                horizontal_lines.add(r)
                break  # Move to next row after finding first non-zero

    # Step 3: Handle intersections
    for r in horizontal_lines:
        for c in vertical_lines:
            if output_grid.values[r][c] != 2:  # If not already marked as intersection
                vertical_color = output_grid.values[0][c]
                horizontal_color = output_grid.values[r][0]
                if vertical_color != horizontal_color:
                    output_grid.values[r][c] = 2

    # Step 4: Preserve original non-zero values
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                output_grid.values[r][c] = input_grid.values[r][c]

    return output_grid
