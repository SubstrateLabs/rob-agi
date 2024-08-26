from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

COMPLEXITY_THRESHOLD = 0.8
GEOMETRIC_THRESHOLD = 1.2
FORM_BONUS = 0.5

def solve_009d5c81(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing the color of the larger shape (color 8)
    based on its geometric properties. The smaller shape (color 1) is removed,
    and the rest of the grid remains black (color 0).

    The new color of the larger shape is determined as follows:
    - Orange (7) if the shape is complex (high perimeter-to-area ratio)
    - Red (2) if the shape has more straight lines, right angles, and grid-like patterns
    - Green (3) if the shape has more curved or organic features

    The function analyzes the shape's complexity, geometric features, and organic features
    to make this determination.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed output grid
    """
    shape = input_grid.find_connected_regions(8)[0]
    analysis_results = analyze_shape(input_grid, shape)
    new_color = determine_color(*analysis_results)
    
    output_grid = ColoredGrid(values=[[0 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    for r, c in shape:
        output_grid.values[r][c] = new_color
    
    return output_grid

def analyze_shape(grid: ColoredGrid, shape: List[Tuple[int, int]]):
    complexity = calculate_complexity(shape)
    straight_score = count_straight_elements(shape)
    curve_score = count_curved_elements(shape)
    grid_score = detect_grid_pattern(shape)
    symmetry_score = measure_symmetry(grid, shape)
    form = identify_form(shape)
    
    return complexity, straight_score, curve_score, grid_score, symmetry_score, form

def determine_color(complexity, straight_score, curve_score, grid_score, symmetry_score, form):
    if complexity > COMPLEXITY_THRESHOLD:
        return 7  # Orange
    
    geometric_score = straight_score + grid_score + symmetry_score
    organic_score = curve_score
    
    if form in ['face', 'natural']:
        organic_score += FORM_BONUS
    elif form in ['geometric', 'grid']:
        geometric_score += FORM_BONUS
    
    if geometric_score > organic_score * GEOMETRIC_THRESHOLD:
        return 2  # Red
    else:
        return 3  # Green

def calculate_complexity(shape: List[Tuple[int, int]]) -> float:
    perimeter = calculate_perimeter(shape)
    area = len(shape)
    return perimeter / (area ** 0.5)  # Normalized complexity measure

def count_straight_elements(shape: List[Tuple[int, int]]) -> float:
    straight_lines = sum(1 for r, c in shape if sum((r+1, c) in shape, (r-1, c) in shape, (r, c+1) in shape, (r, c-1) in shape) == 2)
    right_angles = count_right_angles(shape)
    return (straight_lines + right_angles) / len(shape)

def count_curved_elements(shape: List[Tuple[int, int]]) -> float:
    curves = sum(1 for r, c in shape if sum((r+1, c) in shape, (r-1, c) in shape, (r, c+1) in shape, (r, c-1) in shape) > 2)
    return curves / len(shape)

def detect_grid_pattern(shape: List[Tuple[int, int]]) -> float:
    # Simplified grid pattern detection
    grid_score = 0
    for r, c in shape:
        if all((r+i, c+j) in shape for i, j in [(0,0), (0,1), (1,0), (1,1)]):
            grid_score += 1
    return grid_score / len(shape)

def measure_symmetry(grid: ColoredGrid, shape: List[Tuple[int, int]]) -> float:
    rows, cols = grid.num_rows, grid.num_cols
    horizontal_symmetry = sum(1 for r, c in shape if (rows-1-r, c) in shape)
    vertical_symmetry = sum(1 for r, c in shape if (r, cols-1-c) in shape)
    return (horizontal_symmetry + vertical_symmetry) / (2 * len(shape))

def identify_form(shape: List[Tuple[int, int]]) -> str:
    # Simplified form identification
    if len(shape) < 10:
        return 'small'
    elif detect_grid_pattern(shape) > 0.3:
        return 'grid'
    elif measure_symmetry(ColoredGrid(values=[[]]), shape) > 0.7:
        return 'geometric'
    else:
        return 'natural'

def count_right_angles(shape: List[Tuple[int, int]]) -> int:
    right_angles = 0
    for r, c in shape:
        neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
        if sum((nr, nc) in shape for nr, nc in neighbors) == 2:
            diagonal_neighbors = [(r+1, c+1), (r+1, c-1), (r-1, c+1), (r-1, c-1)]
            if sum((nr, nc) in shape for nr, nc in diagonal_neighbors) == 1:
                right_angles += 1
    return right_angles

def calculate_perimeter(shape: List[Tuple[int, int]]) -> int:
    perimeter = 0
    for r, c in shape:
        neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
        perimeter += 4 - sum((nr, nc) in shape for nr, nc in neighbors)
    return perimeter
