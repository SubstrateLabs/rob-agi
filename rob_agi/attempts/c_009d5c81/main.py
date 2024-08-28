from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import math

COMPLEXITY_THRESHOLD = 1.5
GEOMETRIC_THRESHOLD = 1.2

def solve_009d5c81(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by analyzing the larger shape (color 8) and determining its new color
    based on geometric properties and complexity. The smaller shape (color 1) is removed.

    The new color is determined as follows:
    - Orange (7) if the shape is complex (high perimeter-to-area ratio)
    - Red (2) if the shape has more geometric features (straight lines, right angles, symmetry)
    - Green (3) if the shape has more organic features (curves, irregular segments)

    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed output grid
    """
    shape = input_grid.find_connected_regions(8)[0]
    complexity, geometric_score, organic_score = analyze_shape(input_grid, shape)
    new_color = determine_color(complexity, geometric_score, organic_score)
    
    output_grid = ColoredGrid(values=[[0 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    for r, c in shape:
        output_grid.values[r][c] = new_color
    
    return output_grid

def analyze_shape(grid: ColoredGrid, shape: List[Tuple[int, int]]):
    complexity = calculate_complexity(shape)
    geometric_score = analyze_geometric_features(grid, shape)
    organic_score = analyze_organic_features(shape)
    return complexity, geometric_score, organic_score

def determine_color(complexity: float, geometric_score: float, organic_score: float) -> int:
    if complexity > COMPLEXITY_THRESHOLD:
        return 7  # Orange
    elif geometric_score > organic_score * GEOMETRIC_THRESHOLD:
        return 2  # Red
    else:
        return 3  # Green

def calculate_complexity(shape: List[Tuple[int, int]]) -> float:
    perimeter = calculate_perimeter(shape)
    area = len(shape)
    return perimeter / math.sqrt(area)

def analyze_geometric_features(grid: ColoredGrid, shape: List[Tuple[int, int]]) -> float:
    straight_lines = detect_long_lines(shape)
    right_angles = count_right_angles(shape)
    symmetry = measure_symmetry(grid, shape)
    regular_patterns = detect_regular_patterns(shape)
    return (straight_lines + right_angles + symmetry + regular_patterns) / 4

def analyze_organic_features(shape: List[Tuple[int, int]]) -> float:
    curvature = measure_curvature(shape)
    irregular_segments = detect_irregular_segments(shape)
    branching = analyze_branching(shape)
    return (curvature + irregular_segments + branching) / 3

def calculate_perimeter(shape: List[Tuple[int, int]]) -> int:
    perimeter = 0
    for r, c in shape:
        neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
        perimeter += 4 - sum((nr, nc) in shape for nr, nc in neighbors)
    return perimeter

def detect_long_lines(shape: List[Tuple[int, int]]) -> float:
    # Implement Bresenham's line algorithm to detect longer straight segments
    # Return a normalized score based on the proportion of pixels in long lines
    # Placeholder implementation
    return sum(1 for r, c in shape if sum((r+1, c) in shape, (r-1, c) in shape, (r, c+1) in shape, (r, c-1) in shape) == 2) / len(shape)

def count_right_angles(shape: List[Tuple[int, int]]) -> float:
    right_angles = 0
    for r, c in shape:
        neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
        if sum((nr, nc) in shape for nr, nc in neighbors) == 2:
            diagonal_neighbors = [(r+1, c+1), (r+1, c-1), (r-1, c+1), (r-1, c-1)]
            if sum((nr, nc) in shape for nr, nc in diagonal_neighbors) == 1:
                right_angles += 1
    return right_angles / len(shape)

def measure_symmetry(grid: ColoredGrid, shape: List[Tuple[int, int]]) -> float:
    rows, cols = grid.num_rows, grid.num_cols
    horizontal_symmetry = sum(1 for r, c in shape if (rows-1-r, c) in shape)
    vertical_symmetry = sum(1 for r, c in shape if (r, cols-1-c) in shape)
    return (horizontal_symmetry + vertical_symmetry) / (2 * len(shape))

def detect_regular_patterns(shape: List[Tuple[int, int]]) -> float:
    # Implement a more sophisticated pattern detection algorithm
    # Placeholder implementation
    grid_score = 0
    for r, c in shape:
        if all((r+i, c+j) in shape for i, j in [(0,0), (0,1), (1,0), (1,1)]):
            grid_score += 1
    return grid_score / len(shape)

def measure_curvature(shape: List[Tuple[int, int]]) -> float:
    # Implement a curvature measurement algorithm
    # Placeholder implementation
    return sum(1 for r, c in shape if sum((r+1, c) in shape, (r-1, c) in shape, (r, c+1) in shape, (r, c-1) in shape) > 2) / len(shape)

def detect_irregular_segments(shape: List[Tuple[int, int]]) -> float:
    # Implement an algorithm to detect irregular segments
    # Placeholder implementation
    return 1 - detect_long_lines(shape)

def analyze_branching(shape: List[Tuple[int, int]]) -> float:
    # Implement an algorithm to analyze branching points
    # Placeholder implementation
    branching_points = sum(1 for r, c in shape if sum((r+dr, c+dc) in shape for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]) > 2)
    return branching_points / len(shape)
