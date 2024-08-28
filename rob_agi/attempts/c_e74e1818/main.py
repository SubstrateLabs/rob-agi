from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_e74e1818(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by selectively flipping shapes vertically within their bounds
    to achieve a more balanced and symmetrical composition.
    
    The solution involves the following steps:
    1. Identify distinct shapes in the grid
    2. Determine the vertical trisection points of the grid
    3. Process each shape based on its position in the grid:
       - Shapes in the top third are flipped vertically
       - Shapes in the middle third remain unchanged
       - Shapes in the bottom third are flipped vertically and moved upward
    4. Reconstruct the grid with the transformed shapes while maintaining horizontal positions
    
    This function improves vertical symmetry and balance of the image while preserving
    the horizontal positions and widths of shapes.
    """
    shapes = identify_shapes(input_grid)
    transformed_shapes = transform_shapes(shapes, input_grid.num_rows)
    return reconstruct_grid(transformed_shapes, input_grid)

def identify_shapes(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    shapes = []
    visited = set()
    
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                color = grid.values[r][c]
                shape = []
                stack = [(r, c)]
                
                while stack:
                    curr_r, curr_c = stack.pop()
                    if (curr_r, curr_c) not in visited and grid.values[curr_r][curr_c] == color:
                        visited.add((curr_r, curr_c))
                        shape.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                                stack.append((nr, nc))
                
                shapes.append(shape)
    
    return shapes

def transform_shapes(shapes: List[List[Tuple[int, int]]], grid_height: int) -> List[List[Tuple[int, int]]]:
    transformed_shapes = []
    trisection1 = grid_height // 3
    trisection2 = 2 * grid_height // 3
    
    for shape in shapes:
        min_r = min(r for r, _ in shape)
        max_r = max(r for r, _ in shape)
        shape_center = (min_r + max_r) / 2
        
        if shape_center < trisection1:
            transformed_shape = flip_shape_vertically(shape)
        elif shape_center < trisection2:
            transformed_shape = shape  # No change for middle third
        else:
            transformed_shape = flip_shape_vertically(shape)
            # Move the shape upward
            shift = max_r - trisection2
            transformed_shape = [(r - shift, c) for r, c in transformed_shape]
        
        transformed_shapes.append(transformed_shape)
    
    return transformed_shapes

def flip_shape_vertically(shape: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    min_r = min(r for r, _ in shape)
    max_r = max(r for r, _ in shape)
    return [(max_r - (r - min_r), c) for r, c in shape]

def reconstruct_grid(shapes: List[List[Tuple[int, int]]], original_grid: ColoredGrid) -> ColoredGrid:
    new_grid = [[0 for _ in range(original_grid.num_cols)] for _ in range(original_grid.num_rows)]
    
    # Sort shapes based on their original vertical position (top to bottom)
    shapes.sort(key=lambda shape: min(r for r, _ in shape))
    
    for shape in shapes:
        color = original_grid.values[shape[0][0]][shape[0][1]]  # Get color from original grid
        for r, c in shape:
            if 0 <= r < original_grid.num_rows and 0 <= c < original_grid.num_cols:
                new_grid[r][c] = color
    
    return ColoredGrid(values=new_grid)
