from rob_agi.colored_grid import ColoredGrid
from collections import deque
import math

def solve_477d2879(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding colors based on their numeric value and influence.
    
    1. Initialize an output grid with zeros.
    2. Calculate global influence maps for each color present in the input.
    3. Apply colors based on their influence, from highest to lowest.
    4. Fill remaining black cells with the highest-valued neighbor.
    5. Refine color boundaries for more natural-looking results.
    
    Returns the transformed ColoredGrid.
    """
    rows, cols = input_grid.get_dimensions()
    result = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    def calculate_global_influence(color):
        influence = [[0 for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if input_grid.values[r][c] == color:
                    for i in range(rows):
                        for j in range(cols):
                            distance = math.sqrt((r - i)**2 + (c - j)**2)
                            influence[i][j] = max(influence[i][j], color * math.exp(-distance / color))
        return influence

    # Calculate global influence maps
    influence_maps = {}
    for color in range(9, 0, -1):
        if any(color in row for row in input_grid.values):
            influence_maps[color] = calculate_global_influence(color)

    # Apply colors based on influence
    for color in range(9, 0, -1):
        if color in influence_maps:
            for r in range(rows):
                for c in range(cols):
                    if influence_maps[color][r][c] > 0 and (result.values[r][c] == 0 or 
                                                            influence_maps[color][r][c] > influence_maps[result.values[r][c]][r][c]):
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

    # Refine color boundaries
    for _ in range(2):  # Apply refinement twice for better results
        new_result = result.deep_copy()
        for r in range(rows):
            for c in range(cols):
                neighbors = []
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbors.append(result.values[nr][nc])
                if neighbors:
                    most_common = max(set(neighbors), key=neighbors.count)
                    if neighbors.count(most_common) >= 5:  # If majority of neighbors are a different color
                        new_result.values[r][c] = most_common
        result = new_result

    return result
