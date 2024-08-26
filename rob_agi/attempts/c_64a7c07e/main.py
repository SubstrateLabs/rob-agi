from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_64a7c07e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by shifting shapes horizontally towards the center.
    
    The function identifies connected shapes, calculates their center of mass,
    and shifts them as a whole towards the horizontal center of the grid.
    The vertical positions and internal structure of all shapes are preserved.
    """
    height, width = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    
    def find_shapes() -> List[List[Tuple[int, int]]]:
        shapes = []
        visited = set()
        
        def dfs(r: int, c: int) -> List[Tuple[int, int]]:
            shape = []
            stack = [(r, c)]
            while stack:
                curr_r, curr_c = stack.pop()
                if (curr_r, curr_c) not in visited and input_grid.values[curr_r][curr_c] != 0:
                    visited.add((curr_r, curr_c))
                    shape.append((curr_r, curr_c))
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            stack.append((nr, nc))
            return shape
        
        for r in range(height):
            for c in range(width):
                if (r, c) not in visited and input_grid.values[r][c] != 0:
                    shapes.append(dfs(r, c))
        
        return shapes
    
    shapes = find_shapes()
    grid_center = width // 2
    
    for shape in shapes:
        shape_center = sum(c for _, c in shape) / len(shape)
        shift = round(grid_center - shape_center)
        
        # Sort cells by x-coordinate, left-to-right if shifting right, right-to-left if shifting left
        sorted_shape = sorted(shape, key=lambda x: x[1], reverse=shift < 0)
        
        for r, c in sorted_shape:
            new_c = max(0, min(c + shift, width - 1))
            new_grid.values[r][new_c] = input_grid.values[r][c]
    
    return new_grid
