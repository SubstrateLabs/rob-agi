from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_e1d2900e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Identifies 2x2 red squares.
    2. Adds blue dots to the left of the leftmost red square and to the right of the rightmost red square in each row.
    3. Adds blue dots between adjacent red squares in the same row if there's exactly one cell between them.
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

    # Initialize set to store valid blue dot coordinates
    valid_blue_dots = set()

    # Process rows with red squares
    for row in set(r for r, _ in red_squares):
        row_squares = sorted([c for r, c in red_squares if r == row])
        if row_squares:
            # Add blue dot to the left of the leftmost red square
            if row_squares[0] > 0:
                output_grid.set_cell(row, row_squares[0] - 1, 1)
                valid_blue_dots.add((row, row_squares[0] - 1))
            # Add blue dot to the right of the rightmost red square
            if row_squares[-1] < cols - 2:
                output_grid.set_cell(row, row_squares[-1] + 2, 1)
                valid_blue_dots.add((row, row_squares[-1] + 2))
            # Add blue dots between adjacent red squares
            for i in range(len(row_squares) - 1):
                if row_squares[i+1] - row_squares[i] == 3:
                    output_grid.set_cell(row, row_squares[i] + 2, 1)
                    valid_blue_dots.add((row, row_squares[i] + 2))

    # Add edge blue dots to valid set
    for r in range(rows):
        if output_grid.get_cell(r, 0) == 1:
            valid_blue_dots.add((r, 0))
        if output_grid.get_cell(r, cols-1) == 1:
            valid_blue_dots.add((r, cols-1))
    for c in range(cols):
        if output_grid.get_cell(0, c) == 1:
            valid_blue_dots.add((0, c))
        if output_grid.get_cell(rows-1, c) == 1:
            valid_blue_dots.add((rows-1, c))

    # Remove invalid blue dots
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) == 1 and (r, c) not in valid_blue_dots:
                output_grid.set_cell(r, c, 0)

    return output_grid
