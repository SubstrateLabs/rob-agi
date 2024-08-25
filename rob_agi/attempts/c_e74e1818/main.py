from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e74e1818(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid to achieve vertical symmetry and balance.
    
    The solution involves the following steps:
    1. Identify distinct shapes in the grid
    2. Assess vertical symmetry
    3. Transform shapes to improve symmetry
    4. Balance vertical composition
    5. Adjust for collisions and spacing
    6. Reconstruct the grid with transformed shapes
    
    This function focuses on flipping shapes vertically and adjusting their
    vertical positions to create a more symmetrical and balanced composition.
    """
    # Step 1: Identify distinct shapes
    shapes = identify_shapes(input_grid)
    
    # Step 2 & 3: Assess symmetry and transform shapes
    transformed_shapes = transform_shapes(shapes, input_grid.num_cols)
    
    # Step 4 & 5: Balance composition and adjust for collisions
    balanced_shapes = balance_shapes(transformed_shapes, input_grid.num_rows)
    
    # Step 6: Reconstruct the grid
    output_grid = reconstruct_grid(balanced_shapes, input_grid.num_rows, input_grid.num_cols)
    
    return output_grid

def identify_shapes(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    shapes = []
    visited = set()
    
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                shape = []
                color = grid.values[r][c]
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

def transform_shapes(shapes: List[List[Tuple[int, int]]], num_cols: int) -> List[List[Tuple[int, int]]]:
    transformed = []
    mid = num_cols // 2
    
    for shape in shapes:
        left_side = sum(1 for r, c in shape if c < mid)
        right_side = sum(1 for r, c in shape if c >= mid)
        
        if left_side > right_side:
            transformed.append([(r, num_cols - 1 - c) for r, c in shape])
        else:
            transformed.append(shape)
    
    return transformed

def balance_shapes(shapes: List[List[Tuple[int, int]]], num_rows: int) -> List[List[Tuple[int, int]]]:
    balanced = []
    top = min(min(r for r, _ in shape) for shape in shapes)
    bottom = max(max(r for r, _ in shape) for shape in shapes)
    mid = (top + bottom) // 2
    target_mid = num_rows // 2
    
    for shape in shapes:
        shape_mid = sum(r for r, _ in shape) // len(shape)
        offset = target_mid - mid
        balanced.append([(r + offset, c) for r, c in shape])
    
    return balanced

def reconstruct_grid(shapes: List[List[Tuple[int, int]]], num_rows: int, num_cols: int) -> ColoredGrid:
    new_grid = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
    
    for i, shape in enumerate(shapes):
        color = i + 1  # Assign a unique color to each shape
        for r, c in shape:
            if 0 <= r < num_rows and 0 <= c < num_cols:
                new_grid[r][c] = color
    
    return ColoredGrid(values=new_grid)
