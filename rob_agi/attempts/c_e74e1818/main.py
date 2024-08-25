from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_e74e1818(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by flipping shapes vertically when necessary to achieve their natural orientation.
    
    The solution involves the following steps:
    1. Identify distinct shapes in the grid
    2. Analyze each shape to determine if it needs flipping
    3. Flip shapes vertically if they are not in their natural orientation
    4. Reconstruct the grid with transformed shapes
    
    This function focuses on orienting shapes based on their structure and complexity,
    flipping them to achieve a more stable or natural position while maintaining
    their horizontal position and the overall structure of the image.
    """
    shapes = identify_shapes(input_grid)
    flipped_shapes = analyze_and_flip_shapes(shapes, input_grid.num_rows)
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

def analyze_and_flip_shapes(shapes: Dict[int, List[Tuple[int, int]]], num_rows: int) -> Dict[int, List[Tuple[int, int]]]:
    flipped_shapes = {}
    
    for color, shape in shapes.items():
        min_r = min(r for r, _ in shape)
        max_r = max(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        max_c = max(c for _, c in shape)
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        
        if height <= 2 or width == 1:
            flipped_shapes[color] = shape
        else:
            shape_grid = [[0 for _ in range(width)] for _ in range(height)]
            for r, c in shape:
                shape_grid[r - min_r][c - min_c] = 1
            
            top_half = shape_grid[:height//2]
            bottom_half = shape_grid[height//2:]
            top_count = sum(sum(row) for row in top_half)
            bottom_count = sum(sum(row) for row in bottom_half)
            top_complexity = sum(1 for row in top_half if sum(row) > 0)
            bottom_complexity = sum(1 for row in bottom_half if sum(row) > 0)
            
            if top_count < bottom_count or (top_count == bottom_count and top_complexity < bottom_complexity):
                flipped_shape = [(2 * min_r + max_r - r, c) for r, c in shape]
                flipped_shapes[color] = flipped_shape
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
