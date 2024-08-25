from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_05a7bcf2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid according to the following rules:
    1. Identifies a sky blue (8) barrier or uses the middle of the grid if no barrier exists.
    2. Expands yellow (4) regions downwards/rightwards until hitting the barrier, red, or edge.
    3. Expands red (2) regions upwards/leftwards until hitting the barrier, yellow, or edge.
    4. Fills the top/left section with green (3) where not yellow or sky blue.
    5. Fills remaining empty cells with sky blue (8).

    Args:
    input_grid (ColoredGrid): The input grid to transform.

    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def find_sky_blue_barrier(grid):
        for i in range(rows):
            if all(cell == 8 for cell in grid.values[i]):
                return ('horizontal', i)
        for j in range(cols):
            if all(grid.values[i][j] == 8 for i in range(rows)):
                return ('vertical', j)
        return ('horizontal', rows // 2)  # Default to middle row if no barrier found

    def expand_color(grid, start_row, start_col, color, direction):
        if direction == 'down':
            for r in range(start_row, rows):
                if grid.values[r][start_col] in [8, 2]:  # Stop at sky blue or red
                    break
                grid.values[r][start_col] = color
        elif direction == 'up':
            for r in range(start_row, -1, -1):
                if grid.values[r][start_col] in [8, 4]:  # Stop at sky blue or yellow
                    break
                grid.values[r][start_col] = color
        elif direction == 'right':
            for c in range(start_col, cols):
                if grid.values[start_row][c] in [8, 2]:  # Stop at sky blue or red
                    break
                grid.values[start_row][c] = color
        elif direction == 'left':
            for c in range(start_col, -1, -1):
                if grid.values[start_row][c] in [8, 4]:  # Stop at sky blue or yellow
                    break
                grid.values[start_row][c] = color

    orientation, barrier_pos = find_sky_blue_barrier(grid)

    if orientation == 'horizontal':
        # Expand yellow downwards
        for r in range(barrier_pos):
            for c in range(cols):
                if grid.values[r][c] == 4:
                    expand_color(grid, r, c, 4, 'down')
        
        # Expand red upwards
        for r in range(rows - 1, barrier_pos, -1):
            for c in range(cols):
                if grid.values[r][c] == 2:
                    expand_color(grid, r, c, 2, 'up')
        
        # Fill with green in top half
        for r in range(barrier_pos):
            for c in range(cols):
                if grid.values[r][c] not in [4, 8]:
                    grid.values[r][c] = 3
    else:  # vertical orientation
        # Expand yellow rightwards
        for c in range(barrier_pos):
            for r in range(rows):
                if grid.values[r][c] == 4:
                    expand_color(grid, r, c, 4, 'right')
        
        # Expand red leftwards
        for c in range(cols - 1, barrier_pos, -1):
            for r in range(rows):
                if grid.values[r][c] == 2:
                    expand_color(grid, r, c, 2, 'left')
        
        # Fill with green in left half
        for c in range(barrier_pos):
            for r in range(rows):
                if grid.values[r][c] not in [4, 8]:
                    grid.values[r][c] = 3

    # Fill remaining cells with sky blue
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 0:
                grid.values[r][c] = 8

    return grid
