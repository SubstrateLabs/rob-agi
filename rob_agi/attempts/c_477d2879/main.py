from rob_agi.colored_grid import ColoredGrid
from collections import deque

def solve_477d2879(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding colors based on their numeric value and global influence.
    
    1. Create a deep copy of the input grid.
    2. For each non-black cell, calculate its color influence across the entire grid:
       - Use a modified flood fill algorithm to spread influence in all eight directions.
       - Decrease influence with distance, stop at higher-valued colors or grid boundaries.
    3. Apply the color with the highest influence to each cell.
    4. Fill remaining black cells with the highest-numbered non-black neighbor.
    
    Returns the transformed ColoredGrid.
    """
    rows, cols = input_grid.get_dimensions()
    
    def calculate_influence(start_row, start_col, color):
        influence = [[0 for _ in range(cols)] for _ in range(rows)]
        queue = deque([(start_row, start_col, color)])
        visited = set()

        while queue:
            r, c, strength = queue.popleft()
            if (r, c) in visited or strength <= 0:
                continue
            visited.add((r, c))
            influence[r][c] = max(influence[r][c], strength)

            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if input_grid.values[nr][nc] > color:
                        continue
                    queue.append((nr, nc, strength - 1))

        return influence

    # Calculate global influence for each non-black cell
    sorted_cells = [(input_grid.values[r][c], r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] != 0]
    sorted_cells.sort(reverse=True)
    
    max_influence = [[[0, 0] for _ in range(cols)] for _ in range(rows)]  # [influence, color]
    for color, r, c in sorted_cells:
        influence = calculate_influence(r, c, color)
        for i in range(rows):
            for j in range(cols):
                if influence[i][j] > max_influence[i][j][0]:
                    max_influence[i][j] = [influence[i][j], color]
                elif influence[i][j] == max_influence[i][j][0]:
                    max_influence[i][j][1] = max(max_influence[i][j][1], color)

    # Apply color influence
    result = ColoredGrid(values=[[max_influence[i][j][1] for j in range(cols)] for i in range(rows)])

    # Fill remaining black cells
    def get_highest_neighbor(grid, r, c):
        highest = 0
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                highest = max(highest, grid[nr][nc])
        return highest

    for i in range(rows):
        for j in range(cols):
            if result.values[i][j] == 0:
                result.values[i][j] = get_highest_neighbor(result.values, i, j)

    return result
