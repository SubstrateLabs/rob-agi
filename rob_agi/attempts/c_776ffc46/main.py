from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing blue plus shapes to a target color.
    The target color (red or green) is determined by the color most frequently
    adjacent to gray borders, including diagonally adjacent cells. Only valid
    blue plus shapes (5 pixels in a + configuration) are transformed. Other blue
    shapes and colors remain unchanged. All transformations are applied simultaneously.

    1. Identify all gray borders in the grid.
    2. Count red and green pixels adjacent (including diagonally) to all gray borders collectively.
    3. Determine the target color based on the counts (red if tied).
    4. Find all blue plus shapes in the grid.
    5. Transform these blue plus shapes to the target color.
    6. Return the modified grid.

    If no red or green pixels are found adjacent to gray borders, return the original grid.
    """
    GRAY, BLUE, RED, GREEN = 5, 1, 2, 3
    rows, cols = len(input_grid.values), len(input_grid.values[0])

    def is_valid_cell(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def get_adjacent_cells(r: int, c: int) -> Set[Tuple[int, int]]:
        return {(r+dr, c+dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] 
                if (dr != 0 or dc != 0) and is_valid_cell(r+dr, c+dc)}

    def is_blue_plus(r: int, c: int) -> bool:
        if input_grid.values[r][c] != BLUE:
            return False
        return all(is_valid_cell(r+dr, c+dc) and input_grid.values[r+dr][c+dc] == BLUE 
                   for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)])

    # Step 1 & 2: Identify gray borders and count adjacent colors
    adjacent_cells = set()
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == GRAY:
                adjacent_cells.update(get_adjacent_cells(r, c))

    red_count = sum(1 for r, c in adjacent_cells if input_grid.values[r][c] == RED)
    green_count = sum(1 for r, c in adjacent_cells if input_grid.values[r][c] == GREEN)

    # Step 3: Determine the target color
    if red_count == 0 and green_count == 0:
        return input_grid

    target_color = RED if red_count >= green_count else GREEN

    # Step 4: Find all blue plus shapes
    blue_plus_centers = [(r, c) for r in range(rows) for c in range(cols) if is_blue_plus(r, c)]

    # Step 5: Transform blue plus shapes
    result_grid = [row[:] for row in input_grid.values]
    for r, c in blue_plus_centers:
        for dr, dc in [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]:
            result_grid[r+dr][c+dc] = target_color

    # Step 6: Return the modified grid
    return ColoredGrid(values=result_grid)
