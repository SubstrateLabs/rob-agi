from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_94be5b80(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by converting vertical lines at the top into horseshoe shapes,
    while preserving existing horseshoes. The process involves:
    1. Analyzing the top 3x3 area to extract unique colors.
    2. Creating new horseshoes for each extracted color from top to bottom.
    3. Copying existing horseshoes from the input grid.
    4. Filling remaining space with black (0).
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    def extract_top_colors() -> List[int]:
        top_area = input_grid.extract_subgrid(1, 2, 3, 3)
        unique_colors = []
        for row in top_area.values:
            for color in row:
                if color != 0 and color not in unique_colors:
                    unique_colors.append(color)
        return unique_colors

    def create_horseshoe(color: int, top: int, left: int):
        for r in range(top, top + 3):
            for c in range(left, left + 4):
                if r == top + 1 and left < c < left + 3:
                    output_grid.set_cell(r, c, 0)
                else:
                    output_grid.set_cell(r, c, color)

    def is_horseshoe(top: int, left: int) -> bool:
        if top + 2 >= rows or left + 3 >= cols:
            return False
        color = input_grid.get_cell(top, left)
        pattern = [
            [color, color, color, color],
            [color, 0, 0, color],
            [color, color, color, color]
        ]
        return all(input_grid.get_cell(top + r, left + c) == pattern[r][c]
                   for r in range(3) for c in range(4))

    # Step 1 & 2: Extract top colors and create new horseshoes
    top_colors = extract_top_colors()
    for i, color in enumerate(top_colors):
        create_horseshoe(color, 2 + i * 5, 3)

    # Step 3: Copy existing horseshoes
    for r in range(rows - 2):
        for c in range(cols - 3):
            if is_horseshoe(r, c):
                for dr in range(3):
                    for dc in range(4):
                        output_grid.set_cell(r + dr, c + dc, input_grid.get_cell(r + dr, c + dc))

    return output_grid
