from rob_agi.colored_grid import ColoredGrid
from collections import deque

def solve_477d2879(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding colors based on their numeric value.
    
    1. Create a deep copy of the input grid.
    2. Identify all unique colors, excluding black (0) and blue (1).
    3. Sort colors in ascending order.
    4. For each color, perform a flood fill operation:
       - Expand in all eight directions.
       - Overwrite lower-numbered colors, including blue (1) and black (0).
       - Stop at higher-numbered colors or grid boundaries.
    5. Fill remaining black cells with the lowest-numbered non-black neighbor.
    
    Returns the transformed ColoredGrid.
    """
    result = input_grid.deep_copy()
    rows, cols = result.get_dimensions()
    
    def get_neighbors(r, c):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    yield nr, nc
    
    def flood_fill(color):
        queue = deque([(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == color])
        while queue:
            r, c = queue.popleft()
            for nr, nc in get_neighbors(r, c):
                if result.values[nr][nc] < color:
                    result.values[nr][nc] = color
                    queue.append((nr, nc))
    
    def fill_remaining_black():
        for r in range(rows):
            for c in range(cols):
                if result.values[r][c] == 0:
                    neighbor_colors = [result.values[nr][nc] for nr, nc in get_neighbors(r, c) if result.values[nr][nc] != 0]
                    if neighbor_colors:
                        result.values[r][c] = min(neighbor_colors)
    
    colors = sorted(set(cell for row in input_grid.values for cell in row) - {0, 1})
    
    for color in colors:
        flood_fill(color)
    
    fill_remaining_black()
    
    return result
