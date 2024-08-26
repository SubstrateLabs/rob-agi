from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_e74e1818(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by flipping shapes vertically to achieve a more balanced and symmetrical composition.
    
    The solution involves the following steps:
    1. Identify distinct shapes in the grid
    2. Analyze each shape's characteristics and the overall image composition
    3. Decide whether to flip each shape based on individual and global criteria
    4. Apply the transformations and reconstruct the grid
    5. Verify and optimize the final composition
    
    This function aims to improve vertical symmetry and balance of the image while maintaining
    the horizontal positions and relationships between shapes.
    """
    shapes = identify_shapes(input_grid)
    center_of_mass = calculate_center_of_mass(input_grid)
    initial_symmetry = calculate_vertical_symmetry(input_grid)
    
    flipped_shapes = analyze_and_flip_shapes(shapes, input_grid, center_of_mass, initial_symmetry)
    output_grid = reconstruct_grid(flipped_shapes, input_grid)
    
    if calculate_vertical_symmetry(output_grid) <= initial_symmetry:
        output_grid = optimize_composition(shapes, input_grid, center_of_mass)
    
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

def analyze_and_flip_shapes(shapes: Dict[int, List[Tuple[int, int]]], grid: ColoredGrid, center_of_mass: Tuple[float, float], initial_symmetry: float) -> Dict[int, List[Tuple[int, int]]]:
    flipped_shapes = {}
    
    for color, shape in shapes.items():
        min_r = min(r for r, _ in shape)
        max_r = max(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        max_c = max(c for _, c in shape)
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        
        shape_grid = [[0 for _ in range(width)] for _ in range(height)]
        for r, c in shape:
            shape_grid[r - min_r][c - min_c] = 1
        
        top_half = shape_grid[:height//2]
        bottom_half = shape_grid[height//2:]
        top_count = sum(sum(row) for row in top_half)
        bottom_count = sum(sum(row) for row in bottom_half)
        top_complexity = sum(1 for row in top_half if sum(row) > 0)
        bottom_complexity = sum(1 for row in bottom_half if sum(row) > 0)
        
        should_flip = top_count > bottom_count or (top_count == bottom_count and top_complexity > bottom_complexity)
        
        temp_grid = grid.deep_copy()
        flipped_shape = [(2 * min_r + max_r - r, c) for r, c in shape]
        for r, c in shape:
            temp_grid.values[r][c] = 0
        for r, c in flipped_shape:
            if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols:
                temp_grid.values[r][c] = color
        
        new_symmetry = calculate_vertical_symmetry(temp_grid)
        new_center_of_mass = calculate_center_of_mass(temp_grid)
        
        if should_flip and new_symmetry > initial_symmetry and new_center_of_mass[0] > center_of_mass[0]:
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

def optimize_composition(shapes: Dict[int, List[Tuple[int, int]]], original_grid: ColoredGrid, center_of_mass: Tuple[float, float]) -> ColoredGrid:
    best_grid = original_grid
    best_symmetry = calculate_vertical_symmetry(original_grid)
    
    for color in shapes:
        temp_shapes = shapes.copy()
        temp_shapes[color] = [(2 * min(r for r, _ in shapes[color]) + max(r for r, _ in shapes[color]) - r, c) for r, c in shapes[color]]
        temp_grid = reconstruct_grid(temp_shapes, original_grid)
        temp_symmetry = calculate_vertical_symmetry(temp_grid)
        temp_center_of_mass = calculate_center_of_mass(temp_grid)
        
        if temp_symmetry > best_symmetry or (temp_symmetry == best_symmetry and temp_center_of_mass[0] > center_of_mass[0]):
            best_grid = temp_grid
            best_symmetry = temp_symmetry
    
    return best_grid
