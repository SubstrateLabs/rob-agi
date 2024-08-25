from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_af22c60d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by filling in black (0) areas with patterns
    extended from surrounding non-black cells.

    The solution follows these steps:
    1. Iterate through the grid row by row, from top to bottom.
    2. For each black cell encountered:
       a. Look for a sequence of colors to the left.
       b. If not found, look for a sequence of colors above.
       c. If still not found, find the nearest non-black color.
    3. Fill the current black cell and any contiguous black cells to its right
       with the found color sequence or single color.
    4. Repeat until no black cells remain.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with black areas filled in.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def get_left_sequence(r: int, c: int) -> List[int]:
        sequence = []
        c -= 1
        while c >= 0 and grid.get_cell(r, c) != 0:
            sequence.append(grid.get_cell(r, c))
            c -= 1
        return sequence[::-1]

    def get_top_sequence(r: int, c: int) -> List[int]:
        sequence = []
        r -= 1
        while r >= 0 and grid.get_cell(r, c) != 0:
            sequence.append(grid.get_cell(r, c))
            r -= 1
        return sequence

    def get_nearest_color(r: int, c: int) -> int:
        directions = [(0, 1), (1, 0), (-1, 1), (1, 1), (-1, -1), (1, -1), (-1, 0), (0, -1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) != 0:
                return grid.get_cell(nr, nc)
        return 1  # Default to color 1 if no non-black color found

    def fill_sequence(r: int, c: int, sequence: List[int]):
        if not sequence:
            return
        idx = 0
        while c < cols and grid.get_cell(r, c) == 0:
            grid.set_cell(r, c, sequence[idx])
            idx = (idx + 1) % len(sequence)
            c += 1

    for r in range(rows):
        c = 0
        while c < cols:
            if grid.get_cell(r, c) == 0:
                left_seq = get_left_sequence(r, c)
                if left_seq:
                    fill_sequence(r, c, left_seq)
                else:
                    top_seq = get_top_sequence(r, c)
                    if top_seq:
                        fill_sequence(r, c, top_seq)
                    else:
                        nearest_color = get_nearest_color(r, c)
                        fill_sequence(r, c, [nearest_color])
            c += 1

    return grid
