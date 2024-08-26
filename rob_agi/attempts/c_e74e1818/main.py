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
    3. For each shape, determine if flipping it vertically improves the overall composition
    4. Apply the flips that result in the best overall improvement
    5. Reconstruct the grid with the optimal configuration of flipped shapes
    
    This function aims to improve vertical symmetry and balance of the image while maintaining
    the vertical ordering and horizontal positions of shapes.
    """
    shapes = identify_shapes(input_grid)
    flipped_shapes = {}
    
    for color, shape in shapes.items():
        original_metrics = calculate_shape_metrics(shape)
        flipped_shape = flip_shape_vertically(shape)
        flipped_metrics = calculate_shape_metrics(flipped_shape)
        
        if is_better_shape_configuration(flipped_metrics, original_metrics):
            flipped_shapes[color] = flipped_shape
    
    return reconstruct_grid(shapes, input_grid, flipped_shapes)

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

def calculate_shape_metrics(shape: List[Tuple[int, int]]) -> Dict:
    min_r = min(r for r, _ in shape)
    max_r = max(r for r, _ in shape)
    center_of_mass = sum(r for r, _ in shape) / len(shape)
    weight_distribution = sum(1 for r, _ in shape if r > (min_r + max_r) / 2) / len(shape)
    
    return {
        "center_of_mass": center_of_mass,
        "weight_distribution": weight_distribution,
        "height": max_r - min_r + 1
    }

def is_better_shape_configuration(new_metrics: Dict, original_metrics: Dict) -> bool:
    if new_metrics["weight_distribution"] > original_metrics["weight_distribution"]:
        return True
    elif new_metrics["weight_distribution"] == original_metrics["weight_distribution"]:
        return new_metrics["center_of_mass"] < original_metrics["center_of_mass"]
    return False

def flip_shape_vertically(shape: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    min_r = min(r for r, _ in shape)
    max_r = max(r for r, _ in shape)
    return [(2 * min_r + max_r - r, c) for r, c in shape]

def reconstruct_grid(shapes: Dict[int, List[Tuple[int, int]]], original_grid: ColoredGrid, flipped_shapes: Dict[int, List[Tuple[int, int]]] = {}) -> ColoredGrid:
    new_grid = [[0 for _ in range(original_grid.num_cols)] for _ in range(original_grid.num_rows)]
    
    for color, shape in shapes.items():
        current_shape = flipped_shapes.get(color, shape)
        min_r = min(r for r, _ in current_shape)
        max_r = max(r for r, _ in current_shape)
        min_c = min(c for _, c in current_shape)
        
        for r, c in current_shape:
            new_r = r - min_r
            new_grid[new_r][c] = color
    
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
