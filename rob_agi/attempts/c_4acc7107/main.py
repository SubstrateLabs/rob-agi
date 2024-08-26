from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

from typing import List, Tuple, Dict
from collections import defaultdict, deque

def solve_4acc7107(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following steps:
    1. Identifies and groups non-black shapes by color
    2. Vertically flips all shapes
    3. Moves shapes to the bottom of the grid, prioritizing larger shapes
    4. Splits and reconnects shapes if necessary to fit the space
    5. Maintains relative horizontal positions and color adjacency where possible
    6. Consolidates disconnected shapes of the same color when feasible
    7. Ensures maximum utilization of space at the bottom of the grid
    
    The transformation preserves the overall structure and color distribution of the shapes
    while reorganizing them at the bottom of the grid.
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
    
    def center_of_mass(coords: List[Tuple[int, int]]) -> float:
        return sum(c for _, c in coords) / len(coords)
    
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
    
    def find_bottom_position(component: List[Tuple[int, int]], grid: List[List[int]]) -> int:
        max_row = max(r for r, _ in component)
        for offset in range(rows):
            if all(0 <= r + offset < rows and grid[r + offset][c] == 0 for r, c in component):
                return offset
        return 0
    
    # Step 2: Create a new empty grid
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Step 3: Process each color
    for color, coords in sorted(color_coords.items(), key=lambda x: -len(x[1])):
        flipped_coords = vertical_flip(coords)
        original_com = center_of_mass(coords)
        flipped_com = center_of_mass(flipped_coords)
        
        components = find_connected_components(flipped_coords)
        
        for component in components:
            # Find the lowest possible position
            bottom_offset = find_bottom_position(component, new_grid)
            
            # Adjust horizontal position
            horizontal_shift = round(original_com - flipped_com)
            
            # Place the component in the new grid
            for r, c in component:
                new_r = r + bottom_offset
                new_c = max(0, min(cols - 1, c + horizontal_shift))
                if 0 <= new_r < rows and 0 <= new_c < cols and new_grid[new_r][new_c] == 0:
                    new_grid[new_r][new_c] = color
    
    # Step 4: Final alignment
    for c in range(cols):
        for r in range(rows - 1, 0, -1):
            if new_grid[r][c] == 0 and new_grid[r-1][c] != 0:
                new_grid[r][c], new_grid[r-1][c] = new_grid[r-1][c], new_grid[r][c]
    
    return ColoredGrid(values=new_grid)
