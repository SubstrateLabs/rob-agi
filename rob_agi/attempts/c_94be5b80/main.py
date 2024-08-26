from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_94be5b80(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by converting vertical lines at the top into horseshoe shapes,
    while preserving existing horseshoes. The process involves:
    1. Extracting unique colors from the top area (first 3 rows).
    2. Identifying existing horseshoes in the input grid.
    3. Creating new horseshoes for each extracted color not already present.
    4. Preserving existing horseshoes in their original positions.
    5. Filling remaining space with black (0).
    """
    output_grid = ColoredGrid(values=[[0 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    rows, cols = input_grid.get_dimensions()

    def extract_top_colors() -> List[int]:
        unique_colors = set()
        for r in range(3):
            for c in range(cols):
                color = input_grid.get_cell(r, c)
                if color != 0:
                    unique_colors.add(color)
        return list(unique_colors)

    def create_horseshoe(color: int, top: int, left: int):
        for r in range(top, top + 3):
            for c in range(left, left + 6):
                if r == top + 1 and left + 1 < c < left + 4:
                    output_grid.set_cell(r, c, 0)
                else:
                    output_grid.set_cell(r, c, color)

    def is_horseshoe(grid: ColoredGrid, top: int, left: int) -> bool:
        if top + 2 >= rows or left + 5 >= cols:
            return False
        color = grid.get_cell(top, left)
        pattern = [
            [color, color, color, color, color, color],
            [color, 0, 0, 0, 0, color],
            [color, color, color, color, color, color]
        ]
        return all(grid.get_cell(top + r, left + c) == pattern[r][c]
                   for r in range(3) for c in range(6))

    def find_existing_horseshoes() -> List[Tuple[int, int, int]]:
        horseshoes = []
        for r in range(rows - 2):
            for c in range(cols - 5):
                if is_horseshoe(input_grid, r, c):
                    horseshoes.append((input_grid.get_cell(r, c), r, c))
        return horseshoes

    # Step 1: Extract top colors
    top_colors = extract_top_colors()

    # Step 2: Identify existing horseshoes
    existing_horseshoes = find_existing_horseshoes()
    existing_colors = set(color for color, _, _ in existing_horseshoes)

    # Step 3: Create new horseshoes
    current_row = 5
    for color in top_colors:
        if color not in existing_colors and current_row + 3 <= rows:
            create_horseshoe(color, current_row, 3)
            current_row += 5

    # Step 4: Preserve existing horseshoes
    for color, top, left in existing_horseshoes:
        for r in range(3):
            for c in range(6):
                output_grid.set_cell(top + r, left + c, input_grid.get_cell(top + r, left + c))

    return output_grid
