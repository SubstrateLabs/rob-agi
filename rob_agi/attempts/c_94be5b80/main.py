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
    5. Maintaining a 2-row gap between horseshoes when possible, but allowing flexibility.
    6. Preserving existing horseshoes in their original positions.
    7. Ensuring vertical alignment of all horseshoes.
    8. Filling remaining space with black (0).
    9. Limiting the output grid to a maximum of 30 rows or the input grid height, whichever is larger.
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

    # Sort layout, preserving order of colors from top
    def sort_key(x):
        if x[1] != -1:  # Existing horseshoe
            return (x[1], top_colors.index(x[0]) if x[0] in top_colors else len(top_colors))
        else:  # New horseshoe
            return (float('inf'), top_colors.index(x[0]))

    layout.sort(key=sort_key)

    # Calculate output grid size
    output_rows = max(rows, min(len(layout) * 5, 30))  # Use input height or up to 30 rows
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(output_rows)])

    # Step 4: Determine positions and create horseshoes
    current_row = 0
    for color, top, left in layout:
        if top != -1:  # Existing horseshoe
            create_horseshoe(output_grid, color, top, left)
        else:  # New horseshoe
            create_horseshoe(output_grid, color, current_row, 3)
            current_row += 5

        # Ensure we don't exceed the output grid size
        if current_row >= output_rows:
            break

    # Step 5: Adjust spacing
    horseshoe_rows = [r for r in range(output_rows) if any(output_grid.get_cell(r, c) != 0 for c in range(cols))]
    for i in range(len(horseshoe_rows) - 1):
        gap = horseshoe_rows[i+1] - horseshoe_rows[i] - 3
        if gap > 2:
            # Move the lower horseshoe up
            shift = min(gap - 2, 2)  # Try to get 2-row gap, but don't overlap
            for r in range(horseshoe_rows[i+1], min(horseshoe_rows[i+1]+3, output_rows)):
                for c in range(cols):
                    output_grid.set_cell(r-shift, c, output_grid.get_cell(r, c))
                    output_grid.set_cell(r, c, 0)
            horseshoe_rows[i+1] -= shift

    return output_grid
