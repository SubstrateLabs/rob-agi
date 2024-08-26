from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_94be5b80(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by converting vertical lines at the top into horseshoe shapes,
    while preserving existing horseshoes. The process involves:
    1. Extracting unique colors from the top area (first 3 rows) in order.
    2. Identifying existing horseshoes in the input grid.
    3. Creating a layout plan for all horseshoes (existing and new).
    4. Generating the output grid with horseshoes in their determined positions.
    5. Ensuring exactly 2 rows of space between horseshoes.
    6. Filling remaining space with black (0).
    """
    output_grid = ColoredGrid(values=[[0 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    rows, cols = input_grid.get_dimensions()

    def extract_top_colors() -> List[int]:
        unique_colors = []
        for r in range(3):
            for c in range(cols):
                color = input_grid.get_cell(r, c)
                if color != 0 and color not in unique_colors:
                    unique_colors.append(color)
        return unique_colors

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

    # Step 3: Create layout plan
    layout = existing_horseshoes[:]
    for color in top_colors:
        if color not in existing_colors:
            layout.append((color, -1, -1))  # -1 indicates a new horseshoe
    layout.sort(key=lambda x: x[1])  # Sort by row position

    # Step 4: Determine positions and create horseshoes
    current_row = 5
    for color, top, left in layout:
        if top == -1:  # New horseshoe
            create_horseshoe(color, current_row, 3)
            current_row += 5
        else:  # Existing horseshoe
            create_horseshoe(color, top, left)
            current_row = max(current_row, top + 5)

    return output_grid
