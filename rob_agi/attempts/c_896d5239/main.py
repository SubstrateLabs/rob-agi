from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_896d5239(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by finding rectangular regions around green (3) squares
    and filling them with sky blue (8), while preserving the original green squares.
    
    The algorithm works as follows:
    1. Identify all green squares in the grid.
    2. For each green square, find the largest possible rectangle that can be formed
       around it, containing only black (0), green (3), or already processed sky blue (8) squares.
    3. Sort these rectangles by size (area) in descending order.
    4. Process each rectangle, filling it with sky blue (8) and merging overlapping rectangles.
    5. Restore the original green squares.

    This approach ensures that larger regions are prioritized and the original structure
    of the grid is preserved where necessary.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def find_largest_rectangle(start_x: int, start_y: int) -> Tuple[int, int, int, int]:
        def is_valid(x: int, y: int) -> bool:
            return 0 <= x < rows and 0 <= y < cols and output_grid.get_cell(x, y) in [0, 3, 8]

        max_width = max_height = 1
        while is_valid(start_x, start_y + max_width):
            max_width += 1
        max_width -= 1

        while all(is_valid(start_x + max_height, y) for y in range(start_y, start_y + max_width)):
            max_height += 1
        max_height -= 1

        return start_x, start_y, max_height, max_width

    green_squares = [(r, c) for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) == 3]
    rectangles = [find_largest_rectangle(r, c) for r, c in green_squares]
    rectangles.sort(key=lambda rect: rect[2] * rect[3], reverse=True)

    for x, y, height, width in rectangles:
        for r in range(x, x + height):
            for c in range(y, y + width):
                if output_grid.get_cell(r, c) != 3:
                    output_grid.set_cell(r, c, 8)

    for r, c in green_squares:
        output_grid.set_cell(r, c, 3)

    return output_grid
