from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_e74e1818(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by flipping shapes vertically within their bounds to achieve a more balanced and symmetrical composition.
    
    The solution involves the following steps:
    1. Identify distinct shapes in the grid
    2. Analyze each shape's characteristics and determine if it should be flipped
    3. Iteratively flip shapes and evaluate the overall grid symmetry and balance
    4. Reconstruct the grid with the optimal configuration of flipped shapes
    
    This function aims to improve vertical symmetry and balance of the image while maintaining
    the vertical ordering and horizontal positions of shapes.
    """
    shapes = identify_shapes(input_grid)
    initial_symmetry = calculate_vertical_symmetry(input_grid)
    initial_center_of_mass = calculate_center_of_mass(input_grid)
    
    best_grid = input_grid
    best_symmetry = initial_symmetry
    best_center_of_mass = initial_center_of_mass
    
    for color, shape in shapes.items():
        flipped_shape = flip_shape_vertically(shape)
        temp_grid = reconstruct_grid(shapes, input_grid, {color: flipped_shape})
        temp_symmetry = calculate_vertical_symmetry(temp_grid)
        temp_center_of_mass = calculate_center_of_mass(temp_grid)
        
        if temp_symmetry > best_symmetry or (temp_symmetry == best_symmetry and temp_center_of_mass[0] > best_center_of_mass[0]):
            best_grid = temp_grid
            best_symmetry = temp_symmetry
            best_center_of_mass = temp_center_of_mass
            shapes[color] = flipped_shape
    
    return best_grid

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

def calculate_center_of_mass(grid: ColoredGrid) -> Tuple[float, float]:
    total_mass = sum(sum(row) for row in grid.values)
    if total_mass == 0:
        return grid.num_rows / 2, grid.num_cols / 2
    
    row_sum = sum(r * sum(row) for r, row in enumerate(grid.values))
    col_sum = sum(c * grid.values[r][c] for r in range(grid.num_rows) for c in range(grid.num_cols))
    
    return row_sum / total_mass, col_sum / total_mass

def calculate_vertical_symmetry(grid: ColoredGrid) -> float:
    symmetry_score = 0
    for r in range(grid.num_rows):
        for c in range(grid.num_cols // 2):
            if grid.values[r][c] == grid.values[r][-(c+1)]:
                symmetry_score += 1
    return symmetry_score / (grid.num_rows * grid.num_cols // 2)

def flip_shape_vertically(shape: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    min_r = min(r for r, _ in shape)
    max_r = max(r for r, _ in shape)
    return [(2 * min_r + max_r - r, c) for r, c in shape]

def reconstruct_grid(shapes: Dict[int, List[Tuple[int, int]]], original_grid: ColoredGrid, flipped_shapes: Dict[int, List[Tuple[int, int]]] = {}) -> ColoredGrid:
    new_grid = [[0 for _ in range(original_grid.num_cols)] for _ in range(original_grid.num_rows)]
    
    for color, shape in shapes.items():
        current_shape = flipped_shapes.get(color, shape)
        for r, c in current_shape:
            if 0 <= r < original_grid.num_rows and 0 <= c < original_grid.num_cols:
                new_grid[r][c] = color
    
    return ColoredGrid(values=new_grid)
