from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from itertools import combinations

def solve_e74e1818(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by selectively flipping shapes vertically within their bounds
    to achieve a more balanced and symmetrical composition.
    
    The solution involves the following steps:
    1. Identify distinct shapes in the grid
    2. Analyze each shape's characteristics (weight distribution, bounds)
    3. Generate and evaluate all possible combinations of flipped shapes
    4. Select the best combination based on balance, symmetry, and weight distribution
    5. Reconstruct the grid with the optimal configuration of flipped shapes
    
    This function aims to improve vertical symmetry and balance of the image while maintaining
    the vertical ordering and horizontal positions of shapes.
    """
    shapes = identify_shapes(input_grid)
    initial_metrics = calculate_grid_metrics(input_grid, shapes)
    
    best_grid = input_grid
    best_metrics = initial_metrics
    best_flips = {}
    
    for i in range(len(shapes) + 1):
        for flip_combination in combinations(shapes.keys(), i):
            flipped_shapes = {color: flip_shape_vertically(shape) for color, shape in shapes.items() if color in flip_combination}
            temp_grid = reconstruct_grid(shapes, input_grid, flipped_shapes)
            temp_metrics = calculate_grid_metrics(temp_grid, shapes)
            
            if is_better_configuration(temp_metrics, best_metrics):
                best_grid = temp_grid
                best_metrics = temp_metrics
                best_flips = flipped_shapes
    
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

def calculate_grid_metrics(grid: ColoredGrid, shapes: Dict[int, List[Tuple[int, int]]]) -> Dict:
    center_of_mass = calculate_center_of_mass(grid)
    symmetry = calculate_vertical_symmetry(grid)
    weight_distribution = calculate_weight_distribution(shapes)
    
    return {
        "center_of_mass": center_of_mass,
        "symmetry": symmetry,
        "weight_distribution": weight_distribution
    }

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

def calculate_weight_distribution(shapes: Dict[int, List[Tuple[int, int]]]) -> float:
    total_score = 0
    for shape in shapes.values():
        min_r = min(r for r, _ in shape)
        max_r = max(r for r, _ in shape)
        mid_r = (min_r + max_r) / 2
        lower_half = sum(1 for r, _ in shape if r > mid_r)
        total_score += lower_half / len(shape)
    return total_score / len(shapes)

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

def is_better_configuration(new_metrics: Dict, best_metrics: Dict) -> bool:
    if new_metrics["symmetry"] > best_metrics["symmetry"]:
        return True
    elif new_metrics["symmetry"] == best_metrics["symmetry"]:
        if new_metrics["weight_distribution"] > best_metrics["weight_distribution"]:
            return True
        elif new_metrics["weight_distribution"] == best_metrics["weight_distribution"]:
            return new_metrics["center_of_mass"][0] > best_metrics["center_of_mass"][0]
    return False
