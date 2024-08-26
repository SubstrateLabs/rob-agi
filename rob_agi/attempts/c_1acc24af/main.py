from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_1acc24af(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing rectangular gray (5) regions with both dimensions 2 or greater to red (2).
    The function identifies the largest rectangular subregions within gray areas and changes them to red if they meet the size criteria.
    Non-rectangular gray regions and rectangles smaller than 2x2 remain unchanged.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    processed = set()
    to_be_changed = set()

    def find_largest_rectangle(row: int, col: int) -> Tuple[int, int, int, int]:
        max_width = 0
        while col + max_width < cols and output_grid.get_cell(row, col + max_width) == 5:
            max_width += 1

        max_height = 0
        for width in range(1, max_width + 1):
            height = 0
            while row + height < rows and all(output_grid.get_cell(row + height, col + w) == 5 for w in range(width)):
                height += 1
            if width * height > max_height * max_width:
                max_height = height
                max_width = width

        return row, col, max_height, max_width

    for row in range(rows):
        for col in range(cols):
            if output_grid.get_cell(row, col) == 5 and (row, col) not in processed:
                top, left, height, width = find_largest_rectangle(row, col)
                if height >= 2 and width >= 2:
                    for r in range(top, top + height):
                        for c in range(left, left + width):
                            to_be_changed.add((r, c))
                for r in range(top, top + height):
                    for c in range(left, left + width):
                        processed.add((r, c))

    for row, col in to_be_changed:
        output_grid.set_cell(row, col, 2)

    return output_grid
