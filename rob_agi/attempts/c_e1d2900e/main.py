from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e1d2900e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Identifies 2x2 red squares.
    2. Adds blue dots to the left of the leftmost red square and to the right of the rightmost red square in each row.
    3. Adds blue dots between adjacent red squares in the same row.
    4. Removes isolated blue dots not associated with red squares.
    5. Preserves blue dots on the grid edges.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the rules.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    # Identify 2x2 red squares
    red_squares = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            if all(output_grid.get_cell(r+i, c+j) == 2 for i in range(2) for j in range(2)):
                red_squares.append((r, c))

    # Process rows with red squares
    for row in set(r for r, _ in red_squares):
        row_squares = sorted([c for r, c in red_squares if r == row])
        if row_squares:
            # Add blue dot to the left of the leftmost red square
            if row_squares[0] > 0:
                output_grid.set_cell(row, row_squares[0] - 1, 1)
            # Add blue dot to the right of the rightmost red square
            if row_squares[-1] < cols - 2:
                output_grid.set_cell(row, row_squares[-1] + 2, 1)
            # Add blue dots between adjacent red squares
            for i in range(len(row_squares) - 1):
                if row_squares[i+1] - row_squares[i] == 2:
                    output_grid.set_cell(row, row_squares[i] + 2, 1)

    # Remove isolated blue dots
    def is_valid_blue_dot(r, c):
        if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
            return True
        left_red = c > 0 and all(output_grid.get_cell(r+i, c-1) == 2 for i in range(2))
        right_red = c < cols - 1 and all(output_grid.get_cell(r+i, c+1) == 2 for i in range(2))
        between_red = (c > 0 and c < cols - 1 and
                       all(output_grid.get_cell(r+i, c-2) == 2 for i in range(2)) and
                       all(output_grid.get_cell(r+i, c+2) == 2 for i in range(2)))
        return left_red or right_red or between_red

    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) == 1 and not is_valid_blue_dot(r, c):
                output_grid.set_cell(r, c, 0)

    return output_grid
