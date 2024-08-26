from rob_agi.colored_grid import ColoredGrid

def solve_45bbe264(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by drawing vertical and horizontal lines, and handling intersections.
    
    1. Draw vertical lines for all non-zero colors in the input grid.
    2. Draw horizontal lines for non-zero colors in odd-indexed rows of the input grid.
    3. At intersections of different colors, place a red (2) square.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    # First pass: Draw vertical lines
    for c in range(cols):
        for r in range(rows):
            if input_grid.values[r][c] != 0:
                for row in range(rows):
                    output_grid.values[row][c] = input_grid.values[r][c]
                break  # Move to next column after finding first non-zero

    # Second pass: Draw horizontal lines (only for odd rows)
    for r in range(1, rows, 2):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                for col in range(cols):
                    output_grid.values[r][col] = input_grid.values[r][c]
                break  # Move to next row after finding first non-zero

    # Third pass: Handle intersections
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] != 0:
                if (r > 0 and output_grid.values[r-1][c] != 0 and output_grid.values[r-1][c] != output_grid.values[r][c]) or \
                   (c > 0 and output_grid.values[r][c-1] != 0 and output_grid.values[r][c-1] != output_grid.values[r][c]):
                    output_grid.values[r][c] = 2  # Red for intersection

    return output_grid
