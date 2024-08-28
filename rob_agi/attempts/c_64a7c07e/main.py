from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_64a7c07e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by optimally distributing shapes across the horizontal space.
    
    The function identifies connected shapes, calculates their optimal positions,
    and shifts them horizontally while maintaining their vertical positions and internal structure.
    Shapes are distributed evenly across the available space, preserving their left-to-right order.
    The overall arrangement is centered within the grid.
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
    shapes.sort(key=lambda shape: min(c for _, c in shape))
    
    total_shape_width = sum(max(c for _, c in shape) - min(c for _, c in shape) + 1 for shape in shapes)
    available_space = width - total_shape_width
    ideal_gap = available_space // (len(shapes) + 1) if shapes else 0
    
    # Calculate the starting position to center the entire arrangement
    start_position = (available_space - (len(shapes) - 1) * ideal_gap) // 2
    
    left_boundary = start_position
    for shape in shapes:
        shape_left = min(c for _, c in shape)
        shape_right = max(c for _, c in shape)
        shape_width = shape_right - shape_left + 1
        
        new_left = left_boundary
        
        # Ensure the shape doesn't go beyond the right edge of the grid
        if new_left + shape_width > width:
            new_left = width - shape_width
        
        # Calculate the shift
        shift = new_left - shape_left
        
        # Apply the shift to all cells in the shape
        for r, c in shape:
            new_c = c + shift
            new_grid.values[r][new_c] = input_grid.values[r][c]
        
        # Update the left boundary for the next shape
        left_boundary = new_left + shape_width + ideal_gap
    
    return new_grid
