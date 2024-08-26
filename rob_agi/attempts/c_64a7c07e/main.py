from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_64a7c07e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by shifting non-black shapes horizontally towards the center.
    
    The function identifies connected components (shapes) in the grid, calculates the target column
    for each shape based on the grid width, and shifts each shape independently towards the center.
    The vertical positions and internal structure of each shape are preserved, and the function
    ensures that no part of any shape is pushed out of the grid boundaries.
    """
    height, width = input_grid.get_dimensions()
    
    def find_connected_components() -> List[List[Tuple[int, int]]]:
        visited = set()
        components = []
        
        def dfs(r: int, c: int) -> List[Tuple[int, int]]:
            component = []
            stack = [(r, c)]
            while stack:
                y, x = stack.pop()
                if (y, x) not in visited and input_grid.values[y][x] != 0:
                    visited.add((y, x))
                    component.append((y, x))
                    for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            stack.append((ny, nx))
            return component
        
        for r in range(height):
            for c in range(width):
                if (r, c) not in visited and input_grid.values[r][c] != 0:
                    components.append(dfs(r, c))
        
        return components
    
    components = find_connected_components()
    
    target_column = (width - 1) // 2
    
    new_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    
    for component in components:
        left_x = min(c for _, c in component)
        right_x = max(c for _, c in component)
        shift = target_column - left_x
        
        # Adjust shift if it would push the component out of bounds
        if right_x + shift >= width:
            shift = width - 1 - right_x
        
        for r, c in component:
            new_c = c + shift
            new_grid.values[r][new_c] = input_grid.values[r][c]
    
    return new_grid
