from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e1d2900e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Identifies 2x2 red squares.
    2. Adds exactly two blue dots adjacent to each red square in specific positions.
    3. Removes isolated blue dots not associated with red squares.
    4. Preserves blue dots on the grid edges.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the rules.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def is_red_square(r: int, c: int) -> bool:
        if r + 1 < rows and c + 1 < cols:
            return all(output_grid.get_cell(r+i, c+j) == 2 for i in range(2) for j in range(2))
        return False

    def is_on_edge(r: int, c: int) -> bool:
        return r == 0 or r == rows - 1 or c == 0 or c == cols - 1

    def add_blue_dots(r: int, c: int):
        left_dot = (r, c-1)
        right_dot = (r+1, c+2)
        for dot_r, dot_c in [left_dot, right_dot]:
            if 0 <= dot_r < rows and 0 <= dot_c < cols:
                if output_grid.get_cell(dot_r, dot_c) != 1:
                    output_grid.set_cell(dot_r, dot_c, 1)

    # Step 1: Process red squares
    red_squares = [(r, c) for r in range(rows-1) for c in range(cols-1) if is_red_square(r, c)]
    for r, c in red_squares:
        add_blue_dots(r, c)

    # Step 2 & 3: Remove isolated blue dots, preserve edge dots
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) == 1:
                if not is_on_edge(r, c) and not any(
                    is_red_square(r+i, c+j) and ((r, c) == (r+i, c+j-1) or (r, c) == (r+i+1, c+j+2))
                    for i in [-1, 0] for j in [-1, 0] if 0 <= r+i < rows-1 and 0 <= c+j < cols-1
                ):
                    output_grid.set_cell(r, c, 0)  # Remove isolated dots

    return output_grid
