from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_e74e1818(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by flipping shapes vertically when necessary.
    
    The solution involves the following steps:
    1. Identify distinct shapes in the grid
    2. Analyze each shape to determine if it needs flipping
    3. Flip shapes vertically if needed
    4. Reconstruct the grid with transformed shapes
    
    This function focuses on flipping shapes vertically while maintaining
    their horizontal position and the overall structure of the image.
    """
    # Step 1: Identify distinct shapes
    shapes = identify_shapes(input_grid)
    
    # Step 2 & 3: Analyze shapes and flip if necessary
    flipped_shapes = flip_shapes_if_needed(shapes, input_grid.num_rows)
    
    # Step 4: Reconstruct the grid
    output_grid = reconstruct_grid(flipped_shapes, input_grid)
    
    return output_grid

def identify_shapes(grid: ColoredGrid) -> Dict[int, List[Tuple[int, int]]]:
    shapes = {}
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
                
                shapes[color] = shape
    
    return shapes

def flip_shapes_if_needed(shapes: Dict[int, List[Tuple[int, int]]], num_rows: int) -> Dict[int, List[Tuple[int, int]]]:
    flipped_shapes = {}
    
    for color, shape in shapes.items():
        min_r = min(r for r, _ in shape)
        max_r = max(r for r, _ in shape)
        height = max_r - min_r + 1
        
        if height > 2:  # Only consider flipping shapes taller than 2 cells
            top_half = sum(1 for r, _ in shape if r < (min_r + max_r) // 2)
            bottom_half = sum(1 for r, _ in shape if r > (min_r + max_r) // 2)
            
            if bottom_half > top_half:
                # Flip the shape
                flipped_shape = [(num_rows - 1 - (r - min_r) + min_r, c) for r, c in shape]
                flipped_shapes[color] = flipped_shape
            else:
                flipped_shapes[color] = shape
        else:
            flipped_shapes[color] = shape
    
    return flipped_shapes

def reconstruct_grid(shapes: Dict[int, List[Tuple[int, int]]], original_grid: ColoredGrid) -> ColoredGrid:
    new_grid = [[0 for _ in range(original_grid.num_cols)] for _ in range(original_grid.num_rows)]
    
    for color, shape in shapes.items():
        for r, c in shape:
            if 0 <= r < original_grid.num_rows and 0 <= c < original_grid.num_cols:
                new_grid[r][c] = color
    
    return ColoredGrid(values=new_grid)
