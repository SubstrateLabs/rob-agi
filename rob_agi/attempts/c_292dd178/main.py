from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_292dd178(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling enclosed spaces with red (2).
    
    The function performs the following steps:
    1. Creates a deep copy of the input grid.
    2. Identifies enclosed areas using a flood fill algorithm.
    3. Fills the enclosed areas with red (2).
    4. Ensures that originally blue cells remain blue.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with enclosed areas filled with red.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    def is_enclosed(x: int, y: int) -> bool:
        if grid.values[x][y] == 1:  # Blue cell
            return False
        
        queue = deque([(x, y)])
        visited = set()
        
        while queue:
            cx, cy = queue.popleft()
            
            if cx == 0 or cx == rows - 1 or cy == 0 or cy == cols - 1:
                return False
            
            if (cx, cy) in visited:
                continue
            
            visited.add((cx, cy))
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < rows and 0 <= ny < cols and grid.values[nx][ny] != 1:
                    queue.append((nx, ny))
        
        return True
    
    to_fill = set()
    for x in range(rows):
        for y in range(cols):
            if grid.values[x][y] != 1 and is_enclosed(x, y):
                to_fill.add((x, y))
    
    for x, y in to_fill:
        grid.values[x][y] = 2
    
    return grid
