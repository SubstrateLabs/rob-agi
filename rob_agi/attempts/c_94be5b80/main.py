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
    5. Ensuring exactly 2 rows of space between horseshoes when possible.
    6. Preserving existing horseshoes in their original positions.
    7. Filling remaining space with black (0).
    8. Limiting the output grid to a maximum of 30 rows or the input grid height, whichever is larger.
    """
    rows, cols = input_grid.get_dimensions()

    def extract_top_colors() -> List[int]:
        unique_colors = []
        for r in range(3):
            for c in range(cols):
                color = input_grid.get_cell(r, c)
                if color != 0 and color not in unique_colors:
                    unique_colors.append(color)
        return unique_colors

    def create_horseshoe(grid: ColoredGrid, color: int, top: int, left: int):
        for r in range(top, top + 3):
            for c in range(left, left + 6):
                if r == top + 1 and left + 1 <= c <= left + 4:
                    grid.set_cell(r, c, 0)
                else:
                    grid.set_cell(r, c, color)

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

    # Sort layout, handling colors not in top_colors
    def sort_key(x):
        if x[1] != -1:  # Existing horseshoe
            return (x[1], top_colors.index(x[0]) if x[0] in top_colors else len(top_colors))
        else:  # New horseshoe
            return (float('inf'), top_colors.index(x[0]))

    layout.sort(key=sort_key)

    # Calculate output grid size
    output_rows = min(max(len(layout) * 5 - 2, rows), 30)  # Limit to 30 rows or input height, whichever is larger
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(output_rows)])

    # Step 4: Determine positions and create horseshoes
    current_row = 0
    for color, top, left in layout:
        if top != -1:  # Existing horseshoe
            create_horseshoe(output_grid, color, top, left)
        elif current_row + 5 <= output_rows:  # New horseshoe
            create_horseshoe(output_grid, color, current_row, 3)
            current_row += 5
        else:
            break  # Stop if we can't fit another horseshoe

    # Step 5: Adjust spacing
    horseshoe_rows = [r for r in range(output_rows) if any(output_grid.get_cell(r, c) != 0 for c in range(cols))]
    for i in range(len(horseshoe_rows) - 1):
        if horseshoe_rows[i+1] - horseshoe_rows[i] > 5:
            # Move the lower horseshoe up
            for r in range(horseshoe_rows[i+1], horseshoe_rows[i+1]+3):
                for c in range(cols):
                    output_grid.set_cell(r-2, c, output_grid.get_cell(r, c))
                    output_grid.set_cell(r, c, 0)
            horseshoe_rows[i+1] -= 2

    return output_grid
