from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_62ab2642(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by applying the following rules:
    1. Fill the largest contiguous area of black (0) cells with sky blue (8), starting from either the top-right or bottom-right corner.
    2. Identify small areas of black cells adjacent to gray (5) cells and fill them with orange (7).
    3. Preserve all original gray (5) cells.

    The function uses flood fill for both the sky blue and orange areas, with a comparison to choose the larger sky blue area.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def flood_fill(x: int, y: int, target_color: int, replacement_color: int, count_only: bool = False) -> int:
        if (x < 0 or x >= cols or y < 0 or y >= rows or 
            output_grid.values[y][x] != target_color):
            return 0
        
        count = 1
        if not count_only:
            output_grid.values[y][x] = replacement_color
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            count += flood_fill(x + dx, y + dy, target_color, replacement_color, count_only)
        return count

    # Fill sky blue area
    top_right_count = flood_fill(cols - 1, 0, 0, 8, count_only=True)
    bottom_right_count = flood_fill(cols - 1, rows - 1, 0, 8, count_only=True)
    
    output_grid = input_grid.deep_copy()  # Reset grid
    if top_right_count >= bottom_right_count:
        flood_fill(cols - 1, 0, 0, 8)
    else:
        flood_fill(cols - 1, rows - 1, 0, 8)

    # Identify and fill orange areas
    orange_candidates = set()
    for y in range(rows):
        for x in range(cols):
            if output_grid.values[y][x] == 0:
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < cols and 0 <= ny < rows and
                        output_grid.values[ny][nx] == 5):
                        orange_candidates.add((x, y))
                        break

    for x, y in orange_candidates:
        if output_grid.values[y][x] == 0:
            flood_fill(x, y, 0, 7)

    return output_grid
