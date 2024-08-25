from rob_agi.colored_grid import ColoredGrid

def solve_8597cfd7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by following these steps:
    1. Find the horizontal gray line.
    2. Scan from right to left for vertical lines that:
       - Extend at least one square below the gray line
       - Prioritize the line with the most squares below the gray line
       - In case of a tie, choose the line with the most squares above the gray line
       - If still tied, the rightmost line is chosen (implicit due to right-to-left scan)
    3. Return a 2x2 grid filled with the color of the chosen line, or black if no line is found.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Find the gray line
    gray_line_index = next(r for r in range(rows) if all(input_grid.values[r][c] == 5 for c in range(cols)))
    
    max_squares_below = 0
    max_squares_above = 0
    chosen_color = 0

    # Scan for the target vertical line
    for col in range(cols - 1, -1, -1):  # Scan from right to left
        current_color = input_grid.values[gray_line_index + 1][col]
        if current_color == 0:
            continue

        # Count squares below the gray line
        squares_below = 0
        for row in range(gray_line_index + 1, rows):
            if input_grid.values[row][col] == current_color:
                squares_below += 1
            else:
                break

        # Count squares above the gray line
        squares_above = 0
        for row in range(gray_line_index - 1, -1, -1):
            if input_grid.values[row][col] == current_color:
                squares_above += 1
            else:
                break

        # Update the chosen line if necessary
        if squares_below > max_squares_below or (squares_below == max_squares_below and squares_above > max_squares_above):
            max_squares_below = squares_below
            max_squares_above = squares_above
            chosen_color = current_color

    # Create the output grid
    return ColoredGrid(values=[[chosen_color] * 2 for _ in range(2)])
