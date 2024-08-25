from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_896d5239(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by finding all valid rectangles with green (3) squares at the corners
    and filling them with sky blue (8), while preserving the original green and blue (1) squares.
    
    The algorithm works as follows:
    1. Identify all green squares in the grid.
    2. Generate all valid rectangles with green squares at the corners.
    3. Fill all valid rectangles with sky blue (8).
    4. Restore the original green and blue squares.

    This approach ensures that all valid rectangles are filled with sky blue,
    allowing for overlaps, while maintaining the original pattern of green and blue squares.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    green_squares = [(r, c) for r in range(rows) for c in range(cols) if input_grid.get_cell(r, c) == 3]

    def generate_valid_rectangles():
        valid_rectangles = []
        for i, (r1, c1) in enumerate(green_squares):
            for (r2, c2) in green_squares[i+1:]:
                top_left = (min(r1, r2), min(c1, c2))
                bottom_right = (max(r1, r2), max(c1, c2))
                if (top_left[0], bottom_right[1]) in green_squares and (bottom_right[0], top_left[1]) in green_squares:
                    valid_rectangles.append((top_left, bottom_right))
        return valid_rectangles

    def fill_rectangle(top_left, bottom_right):
        for r in range(top_left[0], bottom_right[0] + 1):
            for c in range(top_left[1], bottom_right[1] + 1):
                if (r, c) not in [(top_left[0], top_left[1]), (top_left[0], bottom_right[1]),
                                  (bottom_right[0], top_left[1]), (bottom_right[0], bottom_right[1])]:
                    output_grid.set_cell(r, c, 8)

    valid_rectangles = generate_valid_rectangles()
    for top_left, bottom_right in valid_rectangles:
        fill_rectangle(top_left, bottom_right)

    # Restore original green and blue squares
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) in [1, 3]:
                output_grid.set_cell(r, c, input_grid.get_cell(r, c))

    return output_grid
