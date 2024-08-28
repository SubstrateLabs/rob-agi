from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing blue plus shapes to a target color.
    The target color (red or green) is determined by the color most frequently
    adjacent to or enclosed by gray borders. Only valid blue plus shapes
    (5 pixels in a + configuration) are transformed. Other blue shapes and
    colors remain unchanged. All transformations are applied simultaneously.

    1. Identify gray borders and determine the target color.
    2. Find all blue plus shapes in the grid.
    3. Transform these blue plus shapes to the target color.
    4. Return the modified grid.
    """
    GRAY, BLUE, RED, GREEN = 5, 1, 2, 3
    rows, cols = len(input_grid.values), len(input_grid.values[0])
    result_grid = [row[:] for row in input_grid.values]

    def is_valid_cell(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def get_adjacent_cells(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (dr != 0 or dc != 0) and is_valid_cell(r+dr, c+dc)]

    def is_blue_plus(r: int, c: int) -> bool:
        if result_grid[r][c] != BLUE:
            return False
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if not is_valid_cell(nr, nc) or result_grid[nr][nc] != BLUE:
                return False
        return True

    gray_cells = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == GRAY]
    
    red_count, green_count = 0, 0
    for gr, gc in gray_cells:
        for nr, nc in get_adjacent_cells(gr, gc):
            if input_grid.values[nr][nc] == RED:
                red_count += 1
            elif input_grid.values[nr][nc] == GREEN:
                green_count += 1

    if red_count == 0 and green_count == 0:
        return input_grid

    target_color = RED if red_count >= green_count else GREEN
    
    blue_plus_shapes = []
    for r in range(rows):
        for c in range(cols):
            if is_blue_plus(r, c):
                blue_plus_shapes.append((r, c))
    
    for r, c in blue_plus_shapes:
        for dr, dc in [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]:
            result_grid[r+dr][c+dc] = target_color
    
    return ColoredGrid(values=result_grid)
