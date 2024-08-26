from rob_agi.colored_grid import ColoredGrid
import random

def find_connected_regions(grid, target):
    rows, cols = len(grid), len(grid[0])
    visited = set()
    regions = []

    def dfs(x, y):
        if (x, y) in visited or x < 0 or x >= rows or y < 0 or y >= cols or grid[x][y] != target:
            return []
        visited.add((x, y))
        region = [(x, y)]
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            region.extend(dfs(x + dx, y + dy))
        return region

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == target and (i, j) not in visited:
                regions.append(dfs(i, j))

    return regions

def assign_color(x, y, current_grid, last_assigned_color):
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    color_counts = {2: 0, 5: 0}
    for nx, ny in neighbors:
        if 0 <= nx < len(current_grid) and 0 <= ny < len(current_grid[0]):
            if current_grid[nx][ny] in [2, 5]:
                color_counts[current_grid[nx][ny]] += 1
    if color_counts[2] > color_counts[5]:
        return 5
    elif color_counts[5] > color_counts[2]:
        return 2
    else:
        return 5 if last_assigned_color == 2 else 2

def color_regions(input_grid, regions):
    output_grid = [[0 for _ in row] for row in input_grid]
    last_assigned_color = random.choice([2, 5])
    for region in regions:
        for x, y in sorted(region):
            if output_grid[x][y] == 0:
                new_color = assign_color(x, y, output_grid, last_assigned_color)
                output_grid[x][y] = new_color
                last_assigned_color = new_color
    return output_grid

def solve_a8610ef7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by replacing sky blue (8) regions with red (2) and gray (5) colors.
    The algorithm finds connected regions of 8s, then assigns 2 and 5 to these regions
    based on their relative positions and neighboring colors. Isolated 8s are colored
    based on their nearest colored neighbors.
    """
    grid_2d = input_grid.values
    
    regions = find_connected_regions(grid_2d, 8)
    
    colored_grid = color_regions(grid_2d, regions)
    
    for x in range(len(grid_2d)):
        for y in range(len(grid_2d[0])):
            if grid_2d[x][y] == 8 and colored_grid[x][y] == 0:
                colored_grid[x][y] = assign_color(x, y, colored_grid, 2)  # Use 2 as default last_assigned_color
    
    return ColoredGrid(values=colored_grid)
