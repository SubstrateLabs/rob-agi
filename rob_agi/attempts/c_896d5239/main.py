from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_896d5239(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by finding rectangular regions with green (3) squares at the corners
    and filling them with sky blue (8), while preserving the original green and blue (1) squares.
    
    The algorithm works as follows:
    1. Identify all green squares in the grid.
    2. Generate all possible rectangles with green squares at the corners.
    3. Sort rectangles by area in descending order.
    4. Apply sky blue rectangles without overlapping, except at the corners.
    5. Preserve the original green and blue squares.

    This approach ensures that larger sky blue regions are prioritized,
    allows for connected sky blue regions, and maintains the original pattern
    of green and blue squares.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    green_squares = [(r, c) for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) == 3]

    def is_valid_rectangle(top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> bool:
        r1, c1 = top_left
        r2, c2 = bottom_right
        return all(
            output_grid.get_cell(r, c) in [0, 3]
            for r in range(r1, r2 + 1)
            for c in range(c1, c2 + 1)
            if (r, c) not in [top_left, (r1, c2), (r2, c1), bottom_right]
        )

    def generate_rectangles() -> List[Tuple[Tuple[int, int], Tuple[int, int], int]]:
        rectangles = []
        for i, (r1, c1) in enumerate(green_squares):
            for r2, c2 in green_squares[i:]:
                if r1 <= r2 and c1 <= c2 and (r1, c1) != (r2, c2):
                    if is_valid_rectangle((r1, c1), (r2, c2)):
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        rectangles.append(((r1, c1), (r2, c2), area))
        return rectangles

    def fill_rectangle(top_left: Tuple[int, int], bottom_right: Tuple[int, int], filled: Set[Tuple[int, int]]):
        r1, c1 = top_left
        r2, c2 = bottom_right
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if (r, c) not in [top_left, (r1, c2), (r2, c1), bottom_right] and (r, c) not in filled:
                    output_grid.set_cell(r, c, 8)
                    filled.add((r, c))

    rectangles = generate_rectangles()
    rectangles.sort(key=lambda x: x[2], reverse=True)

    filled = set()
    for (r1, c1), (r2, c2), _ in rectangles:
        if not any((r, c) in filled for r in range(r1 + 1, r2) for c in range(c1 + 1, c2)):
            fill_rectangle((r1, c1), (r2, c2), filled)

    # Preserve original green and blue squares
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) in [1, 3]:
                output_grid.set_cell(r, c, input_grid.get_cell(r, c))

    return output_grid
