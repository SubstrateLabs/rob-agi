from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_94be5b80(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by converting vertical lines at the top into horseshoe shapes,
    while preserving existing horseshoes. The process involves:
    1. Analyzing the input grid to identify unique colors and their positions.
    2. Identifying existing horseshoes in the input grid.
    3. Calculating available space and adjusting horseshoe parameters if necessary.
    4. Creating new horseshoes for colors that don't have one.
    5. Placing all horseshoes (existing and new) in the output grid.
    6. Maintaining the order of colors from top to bottom.
    7. Adjusting horizontal positioning to center horseshoes if there's excess space.
    8. Ensuring all horseshoes fit within the grid boundaries.
    9. Preserving the input grid dimensions in the output.
    """
    rows, cols = input_grid.get_dimensions()

    def extract_colors() -> List[int]:
        unique_colors = []
        for r in range(rows):
            for c in range(cols):
                color = input_grid.get_cell(r, c)
                if color != 0 and color not in unique_colors:
                    unique_colors.append(color)
        return unique_colors

    def create_horseshoe(grid: ColoredGrid, color: int, top: int, left: int):
        for r in range(top, top + 3):
            for c in range(left, left + 7):
                if r == top + 1 and left + 1 <= c <= left + 5:
                    grid.set_cell(r, c, 0)
                else:
                    grid.set_cell(r, c, color)

    def is_horseshoe(grid: ColoredGrid, top: int, left: int) -> bool:
        if top + 2 >= rows or left + 6 >= cols:
            return False
        color = grid.get_cell(top, left)
        pattern = [
            [color, color, color, color, color, color, color],
            [color, 0, 0, 0, 0, 0, color],
            [color, color, color, color, color, color, color]
        ]
        return all(grid.get_cell(top + r, left + c) == pattern[r][c]
                   for r in range(3) for c in range(7))

    def find_existing_horseshoes() -> List[Tuple[int, int, int]]:
        horseshoes = []
        for r in range(rows - 2):
            for c in range(cols - 6):
                if is_horseshoe(input_grid, r, c):
                    horseshoes.append((input_grid.get_cell(r, c), r, c))
        return horseshoes

    # Step 1: Extract colors
    colors = extract_colors()

    # Step 2: Identify existing horseshoes
    existing_horseshoes = find_existing_horseshoes()
    existing_colors = set(color for color, _, _ in existing_horseshoes)

    # Step 3: Calculate available space and adjust parameters
    new_horseshoes = [color for color in colors if color not in existing_colors]
    total_horseshoes = len(existing_horseshoes) + len(new_horseshoes)
    ideal_space = total_horseshoes * 5 - 2  # 3 rows per horseshoe + 2 rows gap, minus 2 for no gap at the end
    available_space = rows

    if ideal_space > available_space:
        gap = max(0, (available_space - total_horseshoes * 3) // (total_horseshoes - 1))
    else:
        gap = 2

    # Step 4 & 5: Create and place horseshoes
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    current_row = 0

    for color, top, left in existing_horseshoes:
        create_horseshoe(output_grid, color, top, left)

    for color in new_horseshoes:
        if current_row + 3 <= rows:
            create_horseshoe(output_grid, color, current_row, 3)
            current_row += 3 + gap

    # Step 6: Maintain color order (already done by the order of processing)

    # Step 7: Adjust horizontal positioning
    left_margin = (cols - 7) // 2
    if left_margin > 3:
        for r in range(rows):
            row = [output_grid.get_cell(r, c) for c in range(cols)]
            shifted_row = [0] * left_margin + row[3:-3] + [0] * (cols - 7 - left_margin)
            for c in range(cols):
                output_grid.set_cell(r, c, shifted_row[c])

    # Step 8 & 9: Ensure fit within boundaries and preserve dimensions (already done)

    return output_grid
