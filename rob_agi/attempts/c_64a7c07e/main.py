from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_64a7c07e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by shifting shapes and individual cells horizontally towards the center.
    
    The function identifies connected shapes and individual cells, calculates their positions,
    and shifts them towards the horizontal center of the grid.
    The vertical positions and internal structure of all shapes are preserved.
    Shapes and cells maintain their relative order and do not overlap.
    """
    height, width = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    grid_center = width // 2
    
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
    
    # Sort shapes by their leftmost x-coordinate
    shapes.sort(key=lambda shape: min(c for _, c in shape))
    
    left_boundary = 0
    for shape in shapes:
        shape_left = min(c for _, c in shape)
        shape_right = max(c for _, c in shape)
        shape_width = shape_right - shape_left + 1
        
        # Calculate the ideal center position for the shape
        ideal_center = (grid_center + left_boundary) // 2
        
        # Calculate the new left boundary for the shape
        new_left = max(left_boundary, ideal_center - shape_width // 2)
        
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
        left_boundary = new_left + shape_width
    
    return new_grid
