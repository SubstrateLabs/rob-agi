from rob_agi.colored_grid import ColoredGrid

def solve_642248e4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies colored borders (top/bottom or left/right).
    2. For each blue (1) square:
       - Determines its quadrant (top-left, top-right, bottom-left, bottom-right).
       - Checks adjacent cells in a specific order based on the quadrant and influence type.
       - Changes the first black (0) cell found to the appropriate border color.
    3. Preserves original blue squares and border colors.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    # Determine influence type and border colors
    if all(output_grid.values[0][c] != 0 for c in range(cols)) and all(output_grid.values[-1][c] != 0 for c in range(cols)):
        influence = "vertical"
        top_color = output_grid.values[0][0]
        bottom_color = output_grid.values[-1][0]
    else:
        influence = "horizontal"
        left_color = output_grid.values[0][0]
        right_color = output_grid.values[0][-1]

    # Calculate midpoints
    row_midpoint = rows // 2
    col_midpoint = cols // 2

    # Define direction orders for each quadrant
    direction_orders = {
        "vertical": {
            "top-left": [(-1, 0), (-1, -1), (-1, 1)],
            "top-right": [(-1, 0), (-1, 1), (-1, -1)],
            "bottom-left": [(1, 0), (1, -1), (1, 1)],
            "bottom-right": [(1, 0), (1, 1), (1, -1)]
        },
        "horizontal": {
            "top-left": [(0, -1), (-1, -1), (1, -1)],
            "top-right": [(0, 1), (-1, 1), (1, 1)],
            "bottom-left": [(0, -1), (1, -1), (-1, -1)],
            "bottom-right": [(0, 1), (1, 1), (-1, 1)]
        }
    }

    # Process the grid
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] == 1:  # Blue square
                quadrant = ("top" if r < row_midpoint else "bottom") + ("-left" if c < col_midpoint else "-right")
                color = top_color if r < row_midpoint else bottom_color if influence == "vertical" else left_color if c < col_midpoint else right_color
                
                for dr, dc in direction_orders[influence][quadrant]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and output_grid.values[nr][nc] == 0:
                        output_grid.values[nr][nc] = color
                        break

    return output_grid
