from rob_agi.colored_grid import ColoredGrid
from collections import deque

def solve_477d2879(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by simultaneously expanding colors based on their numeric value.
    
    1. Create a deep copy of the input grid.
    2. Identify all unique non-zero, non-blue colors.
    3. Initialize a queue with all initial positions of these colors.
    4. Perform simultaneous breadth-first expansion of colors:
       - Expand in all eight directions.
       - Overwrite lower-numbered colors, including blue (1) and black (0).
       - Higher-numbered colors take precedence when colors meet.
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
    
    colors = set(cell for row in input_grid.values for cell in row if cell > 1)
    queue = deque()
    enqueued = set()
    
    for color in colors:
        for r in range(rows):
            for c in range(cols):
                if input_grid.values[r][c] == color:
                    queue.append((r, c, color))
                    enqueued.add((r, c))
    
    while queue:
        r, c, color = queue.popleft()
        for nr, nc in get_neighbors(r, c):
            if result.values[nr][nc] < color:
                result.values[nr][nc] = color
                if (nr, nc) not in enqueued:
                    queue.append((nr, nc, color))
                    enqueued.add((nr, nc))
    
    for r in range(rows):
        for c in range(cols):
            if result.values[r][c] == 0:
                neighbor_colors = [result.values[nr][nc] for nr, nc in get_neighbors(r, c) if result.values[nr][nc] != 0]
                if neighbor_colors:
                    result.values[r][c] = min(neighbor_colors)
    
    return result
