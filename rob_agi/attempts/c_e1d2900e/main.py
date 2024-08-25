from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e1d2900e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Identifies 2x2 red squares and adds blue dots around them.
    2. Removes isolated blue dots not associated with red squares.
    3. Preserves blue dots near grid edges or part of larger patterns.
    4. Ensures each red square has exactly two associated blue dots.
    5. Handles edge cases and maintains balance in dot placement.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the rules.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def is_red_square(r: int, c: int) -> bool:
        if r + 1 < rows and c + 1 < cols:
            return all(output_grid.get_cell(r+i, c+j) == 2 for i in range(2) for j in range(2))
        return False

    def get_valid_blue_positions(r: int, c: int) -> List[Tuple[int, int]]:
        positions = [(r-1, c-1), (r-1, c+2), (r+2, c-1), (r+2, c+2)]  # Corners
        if r == 0 or r == rows - 2:
            positions.extend([(r, c-1), (r, c+2), (r+1, c-1), (r+1, c+2)])  # Sides
        if c == 0 or c == cols - 2:
            positions.extend([(r-1, c), (r-1, c+1), (r+2, c), (r+2, c+1)])  # Top/Bottom
        return [(r, c) for r, c in positions if 0 <= r < rows and 0 <= c < cols]

    def add_blue_dots(r: int, c: int):
        valid_positions = get_valid_blue_positions(r, c)
        existing_blues = [pos for pos in valid_positions if output_grid.get_cell(*pos) == 1]
        if not existing_blues:
            for pos in valid_positions[:2]:
                output_grid.set_cell(*pos, 1)
        elif len(existing_blues) == 1:
            for pos in valid_positions:
                if pos not in existing_blues:
                    output_grid.set_cell(*pos, 1)
                    break

    # Step 1: Process red squares
    for r in range(rows - 1):
        for c in range(cols - 1):
            if is_red_square(r, c):
                add_blue_dots(r, c)

    # Step 2 & 3: Remove isolated blue dots, preserve edge and pattern dots
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) == 1:
                if not any(is_red_square(r+i, c+j) for i in [-1, 0, 1] for j in [-1, 0, 1] if 0 <= r+i < rows-1 and 0 <= c+j < cols-1):
                    if r < 2 or r >= rows - 2 or c < 2 or c >= cols - 2:
                        continue  # Preserve edge dots
                    output_grid.set_cell(r, c, 0)  # Remove isolated dots

    # Step 4: Balance check
    for r in range(rows - 1):
        for c in range(cols - 1):
            if is_red_square(r, c):
                blue_count = sum(1 for pos in get_valid_blue_positions(r, c) if output_grid.get_cell(*pos) == 1)
                if blue_count > 2:
                    for pos in get_valid_blue_positions(r, c):
                        if output_grid.get_cell(*pos) == 1 and blue_count > 2:
                            output_grid.set_cell(*pos, 0)
                            blue_count -= 1

    return output_grid
