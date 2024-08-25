from rob_agi.colored_grid import ColoredGrid

def solve_642248e4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies colored borders (top/bottom for vertical influence, left/right for horizontal influence).
    2. For each blue (1) square:
       - Determines which half it's in (upper/lower for vertical, left/right for horizontal).
       - Checks adjacent cells in a specific order based on the half and influence type.
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

    # Calculate midpoint
    midpoint = rows // 2 if influence == "vertical" else cols // 2

    # Define direction orders for each half
    direction_orders = {
        "vertical": {
            "upper": [(-1, 0), (-1, -1), (-1, 1)],
            "lower": [(1, 0), (1, -1), (1, 1)]
        },
        "horizontal": {
            "left": [(0, -1), (-1, -1), (1, -1)],
            "right": [(0, 1), (-1, 1), (1, 1)]
        }
    }

    # Process the grid
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] == 1:  # Blue square
                if influence == "vertical":
                    half = "upper" if r < midpoint else "lower"
                    color = top_color if r < midpoint else bottom_color
                else:
                    half = "left" if c < midpoint else "right"
                    color = left_color if c < midpoint else right_color
                
                for dr, dc in direction_orders[influence][half]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and output_grid.values[nr][nc] == 0:
                        output_grid.values[nr][nc] = color
                        break

    return output_grid
