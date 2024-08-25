from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_896d5239(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by finding non-overlapping rectangular regions around green (3) squares
    and filling them with sky blue (8), while preserving the original green squares.
    
    The algorithm works as follows:
    1. Identify all green squares in the grid.
    2. For each green square, find the largest possible rectangle that can be formed
       around it, containing only black (0) and green (3) squares.
    3. Sort these rectangles by size (area) in descending order.
    4. Process each rectangle, filling it with sky blue (8) if it doesn't overlap with already processed areas.
    5. Restore the original green squares.

    This approach ensures that larger regions are prioritized, non-overlapping regions are maintained,
    and the original structure of the grid is preserved where necessary.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    processed = [[False for _ in range(cols)] for _ in range(rows)]

    def find_largest_rectangle(start_x: int, start_y: int) -> Tuple[int, int, int, int, int]:
        max_width = max_height = 0
        for height in range(start_x, rows):
            for width in range(start_y, cols):
                if all(output_grid.get_cell(i, j) in [0, 3] for i in range(start_x, height+1) for j in range(start_y, width+1)):
                    area = (height - start_x + 1) * (width - start_y + 1)
                    if area > max_width * max_height:
                        max_width, max_height = width - start_y + 1, height - start_x + 1
                else:
                    break
        return (start_x, start_y, start_x + max_height - 1, start_y + max_width - 1, max_width * max_height)

    def fill_rectangle(rect: Tuple[int, int, int, int, int]) -> None:
        top_left_x, top_left_y, bottom_right_x, bottom_right_y, _ = rect
        if any(processed[i][j] for i in range(top_left_x, bottom_right_x+1) for j in range(top_left_y, bottom_right_y+1)):
            return
        for i in range(top_left_x, bottom_right_x+1):
            for j in range(top_left_y, bottom_right_y+1):
                if output_grid.get_cell(i, j) != 3:
                    output_grid.set_cell(i, j, 8)
                processed[i][j] = True

    green_squares = [(r, c) for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) == 3]
    rectangles = [find_largest_rectangle(r, c) for r, c in green_squares]
    rectangles.sort(key=lambda rect: rect[4], reverse=True)

    for rect in rectangles:
        fill_rectangle(rect)

    for r, c in green_squares:
        output_grid.set_cell(r, c, 3)

    return output_grid
