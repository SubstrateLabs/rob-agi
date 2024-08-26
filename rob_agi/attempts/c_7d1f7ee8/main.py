from rob_agi.colored_grid import ColoredGrid
from typing import List, Set, Tuple, Dict
from collections import deque

def solve_7d1f7ee8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling the interiors of color regions with their respective colors.
    
    The function works as follows:
    1. Identifies all unique colors in the grid.
    2. Sorts colors based on their "outerness" (proximity to grid edges).
    3. For each color, identifies continuous regions and fills their interiors.
    4. Preserves the original boundary structure and handles nested shapes.
    5. Processes colors from outermost to innermost, overwriting inner content when necessary.
    
    Args:
        input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
        ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                if 0 <= r+dr < rows and 0 <= c+dc < cols]
    
    def find_continuous_region(start_r: int, start_c: int, color: int) -> Set[Tuple[int, int]]:
        region = set()
        queue = deque([(start_r, start_c)])
        while queue:
            r, c = queue.popleft()
            if (r, c) not in region and output_grid.values[r][c] == color:
                region.add((r, c))
                queue.extend(get_neighbors(r, c))
        return region
    
    def get_boundary_cells(region: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        return {(r, c) for r, c in region if any(output_grid.values[nr][nc] != output_grid.values[r][c] 
                                                 for nr, nc in get_neighbors(r, c))}
    
    def fill_enclosed_cells(region: Set[Tuple[int, int]], boundary: Set[Tuple[int, int]], color: int):
        temp_grid = [[0 for _ in range(cols)] for _ in range(rows)]
        for r, c in region:
            temp_grid[r][c] = 1
        
        def flood_fill_temp(start_r: int, start_c: int):
            queue = deque([(start_r, start_c)])
            while queue:
                r, c = queue.popleft()
                if 0 <= r < rows and 0 <= c < cols and temp_grid[r][c] == 0:
                    temp_grid[r][c] = 2
                    output_grid.values[r][c] = color
                    queue.extend(get_neighbors(r, c))
        
        for r, c in boundary:
            for nr, nc in get_neighbors(r, c):
                if (nr, nc) not in region:
                    flood_fill_temp(nr, nc)
    
    # Identify unique colors
    colors = set(output_grid.values[r][c] for r in range(rows) for c in range(cols)) - {0}
    
    # Sort colors based on "outerness"
    def color_outerness(color: int) -> int:
        return min(min(r, rows-1-r) + min(c, cols-1-c) 
                   for r in range(rows) for c in range(cols) if output_grid.values[r][c] == color)
    
    sorted_colors = sorted(colors, key=color_outerness)
    
    # Process each color
    for color in sorted_colors:
        processed_cells = set()
        for r in range(rows):
            for c in range(cols):
                if output_grid.values[r][c] == color and (r, c) not in processed_cells:
                    region = find_continuous_region(r, c, color)
                    boundary = get_boundary_cells(region)
                    fill_enclosed_cells(region, boundary, color)
                    processed_cells.update(region)
    
    return output_grid
