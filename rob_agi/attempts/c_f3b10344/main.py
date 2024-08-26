from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Deque
from collections import deque

def solve_f3b10344(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the f3b10344 challenge by creating a sky blue network that connects and surrounds non-black shapes.
    
    The function creates a mask of non-black cells, expands it, performs a flood fill with sky blue (8),
    restores the original shapes, and ensures all non-black areas are surrounded by sky blue.
    The result is a grid where all non-black shapes are connected and surrounded by a sky blue network.
    """
    if all(cell == 0 for row in input_grid.values for cell in row):
        return input_grid

    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    # Create a mask of non-black cells
    mask = [[cell != 0 for cell in row] for row in grid.values]

    # Expand the mask
    expanded_mask = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if mask[r][c]:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            expanded_mask[nr][nc] = True

    # Perform flood fill with sky blue
    def flood_fill(start_r: int, start_c: int):
        queue = deque([(start_r, start_c)])
        while queue:
            r, c = queue.popleft()
            if 0 <= r < rows and 0 <= c < cols and expanded_mask[r][c] and grid.values[r][c] == 0:
                grid.values[r][c] = 8
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        queue.append((r + dr, c + dc))

    # Find the first True cell in the expanded mask and start flood fill
    for r in range(rows):
        for c in range(cols):
            if expanded_mask[r][c]:
                flood_fill(r, c)
                break
        else:
            continue
        break

    # Restore original shapes
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                grid.values[r][c] = input_grid.values[r][c]

    # Final pass to ensure connectivity
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] in [8] + list(range(1, 10)):
                            grid.values[r][c] = 8
                            break
                    else:
                        continue
                    break

    return grid
