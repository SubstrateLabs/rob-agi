from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e1d2900e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Identifies 2x2 red squares and adds exactly two blue dots adjacent to each.
    2. Removes isolated blue dots not associated with red squares.
    3. Preserves blue dots on the grid edges.
    4. Ensures each red square has exactly two adjacent blue dots in a specific pattern.

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

    def get_adjacent_cells(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+i, c+j) for i in [-1, 0, 1] for j in [-1, 0, 1] 
                if 0 <= r+i < rows and 0 <= c+j < cols and (i != 0 or j != 0)]

    def add_blue_dots(r: int, c: int):
        adjacent = get_adjacent_cells(r, c)
        blue_dots = [pos for pos in adjacent if output_grid.get_cell(*pos) == 1]
        
        if len(blue_dots) < 2:
            # Prefer top or left side, then opposite corner
            preferred = [(r-1, c), (r, c-1), (r+1, c+1), (r-1, c+1), (r+1, c-1)]
            for pos in preferred:
                if pos in adjacent and output_grid.get_cell(*pos) != 1:
                    output_grid.set_cell(*pos, 1)
                    blue_dots.append(pos)
                    if len(blue_dots) == 2:
                        break
        
        # Remove extra blue dots
        for pos in blue_dots[2:]:
            output_grid.set_cell(*pos, 0)

    # Step 1: Process red squares
    red_squares = [(r, c) for r in range(rows-1) for c in range(cols-1) if is_red_square(r, c)]
    for r, c in red_squares:
        add_blue_dots(r, c)

    # Step 2 & 3: Remove isolated blue dots, preserve edge dots
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) == 1:
                if not any(is_red_square(r+i, c+j) for i, j in [(0,0), (-1,0), (0,-1), (-1,-1)] 
                           if 0 <= r+i < rows-1 and 0 <= c+j < cols-1):
                    if r == 0 or r == rows-1 or c == 0 or c == cols-1:
                        continue  # Preserve edge dots
                    output_grid.set_cell(r, c, 0)  # Remove isolated dots

    return output_grid
