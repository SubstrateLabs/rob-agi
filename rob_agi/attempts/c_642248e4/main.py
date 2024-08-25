from rob_agi.colored_grid import ColoredGrid

def solve_642248e4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies colored borders (top/bottom or left/right).
    2. For each blue (1) square:
       - If using vertical influence (top/bottom borders):
         - If in the top half, changes the square above to the top border color if it's black (0).
         - If in the bottom half, changes the square below to the bottom border color if it's black (0).
       - If using horizontal influence (left/right borders):
         - If in the left half, changes the square to the left to the left border color if it's black (0).
         - If in the right half, changes the square to the right to the right border color if it's black (0).
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

    # Process the grid
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if output_grid.values[r][c] == 1:  # Blue square
                if influence == "vertical":
                    if r < row_midpoint and output_grid.values[r-1][c] == 0:
                        output_grid.values[r-1][c] = top_color
                    elif r >= row_midpoint and output_grid.values[r+1][c] == 0:
                        output_grid.values[r+1][c] = bottom_color
                else:  # horizontal influence
                    if c < col_midpoint and output_grid.values[r][c-1] == 0:
                        output_grid.values[r][c-1] = left_color
                    elif c >= col_midpoint and output_grid.values[r][c+1] == 0:
                        output_grid.values[r][c+1] = right_color

    return output_grid
