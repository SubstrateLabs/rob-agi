from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict, deque

def solve_4acc7107(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following steps:
    1. Identifies and groups non-black shapes by color
    2. Vertically flips all shapes
    3. Moves shapes to the bottom of the grid, prioritizing larger shapes
    4. Reorganizes and consolidates disconnected shapes of the same color
    5. Maintains relative horizontal positions of colors while allowing flexibility
    6. Ensures maximum utilization of space at the bottom of the grid
    
    The transformation preserves the overall color distribution and general left-to-right order
    while reorganizing shapes at the bottom of the grid for better space utilization.
    """
    rows, cols = input_grid.get_dimensions()
    color_coords = defaultdict(list)
    
    # Step 1: Analyze the input grid
    for r in range(rows):
        for c in range(cols):
            color = input_grid.get_cell(r, c)
            if color != 0:
                color_coords[color].append((r, c))
    
    # Helper functions
    def vertical_flip(coords: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return [(rows - 1 - r, c) for r, c in coords]
    
    def find_connected_components(coords: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        components = []
        visited = set()
        for coord in coords:
            if coord not in visited:
                component = []
                queue = deque([coord])
                while queue:
                    r, c = queue.popleft()
                    if (r, c) not in visited:
                        visited.add((r, c))
                        component.append((r, c))
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = r + dr, c + dc
                            if (nr, nc) in coords and (nr, nc) not in visited:
                                queue.append((nr, nc))
                components.append(component)
        return sorted(components, key=len, reverse=True)
    
    def find_bottom_position(component: List[Tuple[int, int]], grid: List[List[int]], color: int) -> Tuple[int, int]:
        min_col = min(c for _, c in component)
        max_col = max(c for _, c in component)
        for offset in range(rows):
            for shift in range(-min_col, cols - max_col):
                if all(0 <= r + offset < rows and 0 <= c + shift < cols and 
                       (grid[r + offset][c + shift] == 0 or grid[r + offset][c + shift] == color)
                       for r, c in component):
                    return offset, shift
        return 0, 0
    
    # Step 2: Create a new empty grid
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Step 3: Process each color
    for color, coords in sorted(color_coords.items(), key=lambda x: -len(x[1])):
        flipped_coords = vertical_flip(coords)
        components = find_connected_components(flipped_coords)
        
        for component in components:
            # Find the lowest possible position
            bottom_offset, horizontal_shift = find_bottom_position(component, new_grid, color)
            
            # Place the component in the new grid
            for r, c in component:
                new_r = r + bottom_offset
                new_c = c + horizontal_shift
                if 0 <= new_r < rows and 0 <= new_c < cols:
                    new_grid[new_r][new_c] = color
    
    # Step 4: Final alignment
    for c in range(cols):
        for r in range(rows - 1, 0, -1):
            if new_grid[r][c] == 0 and new_grid[r-1][c] != 0:
                new_grid[r][c], new_grid[r-1][c] = new_grid[r-1][c], new_grid[r][c]
    
    return ColoredGrid(values=new_grid)
