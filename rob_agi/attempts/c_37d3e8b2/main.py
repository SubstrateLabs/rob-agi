from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def flood_fill(grid: List[List[int]], x: int, y: int, visited: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    rows, cols = len(grid), len(grid[0])
    stack = [(x, y)]
    region = []
    while stack:
        cx, cy = stack.pop()
        if (cx, cy) not in visited and grid[cx][cy] == 8:
            visited.add((cx, cy))
            region.append((cx, cy))
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    stack.append((nx, ny))
    return region

def get_adjacent_colors(grid: List[List[int]], region: List[Tuple[int, int]]) -> Set[int]:
    rows, cols = len(grid), len(grid[0])
    adjacent_colors = set()
    for x, y in region:
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] not in [0, 8]:
                adjacent_colors.add(grid[nx][ny])
    return adjacent_colors

def solve_37d3e8b2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by identifying connected regions of sky color (8)
    and assigning them new colors based on adjacency and a specific color sequence.
    
    The solution follows these steps:
    1. Create a deep copy of the input grid.
    2. Initialize the color sequence [1, 2, 3, 7].
    3. Traverse the grid from top-left to bottom-right.
    4. For each sky blue (8) cell encountered:
       a. Use flood fill to identify the connected region.
       b. Determine adjacent colors to the region.
       c. Select a color from the sequence that's not adjacent.
       d. Color the entire region with the selected color.
    5. Return the transformed grid.
    """
    grid = input_grid.deep_copy()
    color_sequence = [1, 2, 3, 7]
    color_index = 0
    visited = set()

    rows, cols = len(grid.values), len(grid.values[0])

    for x in range(rows):
        for y in range(cols):
            if grid.values[x][y] == 8 and (x, y) not in visited:
                region = flood_fill(grid.values, x, y, visited)
                adjacent_colors = get_adjacent_colors(grid.values, region)
                
                # Select color
                for _ in range(len(color_sequence)):
                    if color_sequence[color_index] not in adjacent_colors:
                        break
                    color_index = (color_index + 1) % len(color_sequence)
                
                # Color the region
                for rx, ry in region:
                    grid.values[rx][ry] = color_sequence[color_index]
                
                # Move to next color
                color_index = (color_index + 1) % len(color_sequence)

    return grid
