from rob_agi.colored_grid import ColoredGrid
from collections import deque

def solve_477d2879(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding colors based on their numeric value and influence.
    
    1. Create a deep copy of the input grid.
    2. For each non-black cell, calculate its color influence:
       - Expand in all eight directions until blocked by a higher-numbered color or grid boundary.
       - Higher-numbered colors take precedence and contain lower-numbered colors.
    3. Update the grid with the calculated color influences.
    4. Fill remaining black cells with the highest-numbered non-black neighbor.
    
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
    
    def calculate_color_influence(r, c, color):
        queue = deque([(r, c)])
        visited = set([(r, c)])
        while queue:
            cr, cc = queue.popleft()
            for nr, nc in get_neighbors(cr, cc):
                if (nr, nc) not in visited:
                    if result.values[nr][nc] > color:
                        return result.values[nr][nc]
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return color
    
    # Calculate and update color influences
    for r in range(rows):
        for c in range(cols):
            if result.values[r][c] != 0:
                result.values[r][c] = calculate_color_influence(r, c, result.values[r][c])
    
    # Fill remaining black cells
    for r in range(rows):
        for c in range(cols):
            if result.values[r][c] == 0:
                neighbor_colors = [result.values[nr][nc] for nr, nc in get_neighbors(r, c) if result.values[nr][nc] != 0]
                if neighbor_colors:
                    result.values[r][c] = max(neighbor_colors)
    
    return result
