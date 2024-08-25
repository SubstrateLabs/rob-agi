from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict

def find_connected_regions(grid: List[List[int]]) -> List[List[Tuple[int, int]]]:
    rows, cols = len(grid), len(grid[0])
    visited = set()
    regions = []

    def dfs(x: int, y: int) -> List[Tuple[int, int]]:
        stack = [(x, y)]
        region = []
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) not in visited and grid[cx][cy] == 8:
                visited.add((cx, cy))
                region.append((cx, cy))
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < rows and 0 <= ny < cols:
                        stack.append((nx, ny))
        return region

    for x in range(rows):
        for y in range(cols):
            if grid[x][y] == 8 and (x, y) not in visited:
                regions.append(dfs(x, y))

    return sorted(regions, key=lambda r: (-len(r), r[0]))

def get_adjacent_colors(grid: List[List[int]], region: List[Tuple[int, int]]) -> Set[int]:
    rows, cols = len(grid), len(grid[0])
    adjacent_colors = set()
    for x, y in region:
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] not in [0, 8]:
                adjacent_colors.add(grid[nx][ny])
    return adjacent_colors

def solve_37d3e8b2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by identifying connected regions of sky color (8)
    and assigning them new colors based on size, adjacency, and a specific color sequence.
    
    The solution follows these steps:
    1. Create a deep copy of the input grid.
    2. Identify all distinct sky blue (8) regions using a depth-first search.
    3. Sort regions by size (descending) and then by top-left coordinate.
    4. Initialize color palettes: primary [1, 2, 3, 7] and secondary [4, 5, 6].
    5. For each region, choose a color based on adjacent colors and usage frequency.
    6. Fill each region with its chosen color.
    7. Handle any remaining sky blue cells.
    8. Return the transformed grid.
    """
    grid = input_grid.deep_copy()
    regions = find_connected_regions(grid.values)
    
    primary_colors = [1, 2, 3, 7]
    secondary_colors = [4, 5, 6]
    color_usage = {color: 0 for color in primary_colors + secondary_colors}

    def choose_color(adjacent_colors: Set[int]) -> int:
        available_colors = [c for c in primary_colors if c not in adjacent_colors]
        if not available_colors:
            available_colors = [c for c in secondary_colors if c not in adjacent_colors]
        if not available_colors:
            available_colors = primary_colors
        return min(available_colors, key=lambda c: color_usage[c])

    for region in regions:
        adjacent_colors = get_adjacent_colors(grid.values, region)
        color = choose_color(adjacent_colors)
        color_usage[color] += 1
        for x, y in region:
            grid.values[x][y] = color

    # Handle any remaining sky blue cells
    for x in range(len(grid.values)):
        for y in range(len(grid.values[0])):
            if grid.values[x][y] == 8:
                grid.values[x][y] = min(color_usage, key=color_usage.get)

    return grid
