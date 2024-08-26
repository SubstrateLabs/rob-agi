from rob_agi.colored_grid import ColoredGrid
from collections import deque
import math

def solve_477d2879(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding colors based on their numeric value and influence.
    
    1. Initialize an output grid with zeros.
    2. Process colors from highest (9) to lowest (1):
       - For each cell of the current color in the input grid:
         * Calculate an influence map for the entire grid.
         * Apply the color to cells where its influence is highest and greater than existing color.
    3. Fill any remaining black cells with the highest-numbered non-black neighbor.
    
    Returns the transformed ColoredGrid.
    """
    rows, cols = input_grid.get_dimensions()
    result = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    def calculate_influence(start_row, start_col, color):
        influence = [[0 for _ in range(cols)] for _ in range(rows)]
        queue = deque([(start_row, start_col, color)])
        visited = set()
        
        while queue:
            r, c, score = queue.popleft()
            if (r, c) in visited or score <= 0:
                continue
            visited.add((r, c))
            influence[r][c] = max(influence[r][c], score)
            
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    new_score = score - (1 if dr == 0 or dc == 0 else 1.4)  # Less influence diagonally
                    queue.append((nr, nc, new_score))
        
        return influence

    # Process colors from highest to lowest
    for color in range(9, 0, -1):
        influence_map = [[0 for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if input_grid.values[r][c] == color:
                    cell_influence = calculate_influence(r, c, color)
                    for i in range(rows):
                        for j in range(cols):
                            influence_map[i][j] = max(influence_map[i][j], cell_influence[i][j])
        
        for r in range(rows):
            for c in range(cols):
                if influence_map[r][c] > 0 and result.values[r][c] < color:
                    result.values[r][c] = color

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
