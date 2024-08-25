from rob_agi.colored_grid import ColoredGrid
from collections import deque

def solve_477d2879(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding colors based on their numeric value and global influence.
    
    1. Create a new grid of the same dimensions, initialized with zeros.
    2. Process colors from highest (9) to lowest (1):
       - For each cell of the current color in the input grid:
         * Use a flood fill algorithm to expand the color in all eight directions.
         * Fill cells that are either empty (0) or contain a lower-numbered color.
         * Stop at higher-numbered colors or grid boundaries.
    3. Fill any remaining black cells with the highest-numbered non-black neighbor.
    
    Returns the transformed ColoredGrid.
    """
    rows, cols = input_grid.get_dimensions()
    result = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    def flood_fill(start_row, start_col, color):
        queue = deque([(start_row, start_col)])
        while queue:
            r, c = queue.popleft()
            if result.values[r][c] > color:
                continue
            result.values[r][c] = color
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and result.values[nr][nc] < color:
                    queue.append((nr, nc))

    # Process colors from highest to lowest
    for color in range(9, 0, -1):
        for r in range(rows):
            for c in range(cols):
                if input_grid.values[r][c] == color:
                    flood_fill(r, c, color)

    # Fill remaining black cells
    def get_highest_neighbor(r, c):
        highest = 0
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                highest = max(highest, result.values[nr][nc])
        return highest

    changed = True
    while changed:
        changed = False
        for r in range(rows):
            for c in range(cols):
                if result.values[r][c] == 0:
                    highest = get_highest_neighbor(r, c)
                    if highest > 0:
                        result.values[r][c] = highest
                        changed = True

    return result
