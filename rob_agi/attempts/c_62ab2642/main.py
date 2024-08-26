from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_62ab2642(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by applying the following rules:
    1. Fill the largest corner-connected area of black (0) cells with sky blue (8), starting from either the top-right or bottom-right corner.
    2. Identify isolated black areas completely surrounded by gray (5) cells or the grid edge and fill them with orange (7).
    3. Preserve all original gray (5) cells.
    4. Leave any remaining black (0) cells unchanged.

    The function uses an iterative flood fill for both the sky blue and orange areas, with a comparison to choose the larger sky blue area.
    Isolated black areas are identified by checking if they are completely surrounded by gray cells or the grid edge.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def flood_fill(x: int, y: int, target_color: int, replacement_color: int, count_only: bool = False):
        if output_grid.values[y][x] != target_color:
            return 0, set()
        
        stack = [(x, y)]
        count = 0
        filled = set()
        
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in filled:
                continue
            
            if output_grid.values[cy][cx] == target_color:
                count += 1
                filled.add((cx, cy))
                if not count_only:
                    output_grid.values[cy][cx] = replacement_color
                
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < cols and 0 <= ny < rows and output_grid.values[ny][nx] == target_color:
                        stack.append((nx, ny))
        
        return count, filled

    # Determine and fill largest corner-connected area
    top_right_count, _ = flood_fill(cols-1, 0, 0, 8, count_only=True)
    bottom_right_count, _ = flood_fill(cols-1, rows-1, 0, 8, count_only=True)

    if top_right_count >= bottom_right_count:
        flood_fill(cols-1, 0, 0, 8)
    else:
        flood_fill(cols-1, rows-1, 0, 8)

    # Identify and fill isolated black areas
    def is_surrounded_by_gray(x, y):
        if output_grid.values[y][x] != 0:
            return False
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if output_grid.values[ny][nx] not in [5]:
                    return False
            # Edge of grid counts as gray
        return True

    for y in range(rows):
        for x in range(cols):
            if is_surrounded_by_gray(x, y):
                flood_fill(x, y, 0, 7)

    return output_grid
