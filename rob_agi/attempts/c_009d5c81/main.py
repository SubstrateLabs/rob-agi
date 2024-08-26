from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_009d5c81(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing the color of the larger shape (color 8)
    based on its geometric properties. The smaller shape (color 1) is removed,
    and the rest of the grid remains black (color 0).

    The new color of the larger shape is determined as follows:
    - Orange (7) if the shape is complex (high perimeter-to-area ratio)
    - Red (2) if the shape has more straight lines and right angles
    - Green (3) if the shape has more curved or organic features

    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed output grid
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Extract the larger shape (color 8)
    larger_shape = input_grid.find_connected_regions(8)[0]

    # Analyze the larger shape
    complexity = analyze_complexity(larger_shape)
    mechanical_score = analyze_mechanical_features(larger_shape)
    natural_score = analyze_natural_features(larger_shape)

    # Determine the new color
    if complexity > 0.8:  # Adjusted threshold for complexity
        new_color = 7  # Orange
    elif mechanical_score > natural_score:
        new_color = 2  # Red
    else:
        new_color = 3  # Green

    # Transform the grid
    for r, c in larger_shape:
        output_grid.values[r][c] = new_color

    return output_grid

def analyze_complexity(shape: List[Tuple[int, int]]) -> float:
    perimeter = calculate_perimeter(shape)
    area = len(shape)
    return perimeter / (area ** 0.5)  # Normalized complexity measure

def analyze_mechanical_features(shape: List[Tuple[int, int]]) -> float:
    straight_lines = sum(1 for r, c in shape if sum((r+1, c) in shape, (r-1, c) in shape, (r, c+1) in shape, (r, c-1) in shape) == 2)
    right_angles = count_right_angles(shape)
    return (straight_lines + right_angles) / len(shape)

def analyze_natural_features(shape: List[Tuple[int, int]]) -> float:
    curves = sum(1 for r, c in shape if sum((r+1, c) in shape, (r-1, c) in shape, (r, c+1) in shape, (r, c-1) in shape) > 2)
    return curves / len(shape)

def count_right_angles(shape: List[Tuple[int, int]]) -> int:
    right_angles = 0
    for r, c in shape:
        neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
        if sum((nr, nc) in shape for nr, nc in neighbors) == 2:
            diagonal_neighbors = [(r+1, c+1), (r+1, c-1), (r-1, c+1), (r-1, c-1)]
            if sum((nr, nc) in shape for nr, nc in diagonal_neighbors) == 1:
                right_angles += 1
    return right_angles

def analyze_mechanical_features(grid: ColoredGrid, shape: List[Tuple[int, int]]) -> float:
    lines = grid.detect_lines()
    straight_lines = sum(1 for line in lines if len(line[1]) > 3)
    right_angles = count_right_angles(shape)
    symmetry = measure_symmetry(grid, shape)
    return straight_lines + right_angles + symmetry

def analyze_natural_features(grid: ColoredGrid, shape: List[Tuple[int, int]]) -> float:
    curves = count_curves(grid, shape)
    irregularity = measure_irregularity(shape)
    return curves + irregularity

def analyze_complexity(grid: ColoredGrid, shape: List[Tuple[int, int]]) -> float:
    perimeter = calculate_perimeter(shape)
    area = len(shape)
    return perimeter / (area ** 0.5)  # Normalized complexity measure

def count_right_angles(shape: List[Tuple[int, int]]) -> int:
    # Simplified right angle detection
    right_angles = 0
    for r, c in shape:
        neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
        if sum((nr, nc) in shape for nr, nc in neighbors) == 2:
            right_angles += 1
    return right_angles

def measure_symmetry(grid: ColoredGrid, shape: List[Tuple[int, int]]) -> float:
    # Simplified symmetry measure
    rows, cols = grid.get_dimensions()
    horizontal_symmetry = sum(1 for r, c in shape if (rows-1-r, c) in shape)
    vertical_symmetry = sum(1 for r, c in shape if (r, cols-1-c) in shape)
    return (horizontal_symmetry + vertical_symmetry) / len(shape)

def count_curves(grid: ColoredGrid, shape: List[Tuple[int, int]]) -> int:
    # Simplified curve detection
    curves = 0
    for r, c in shape:
        neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
        if 2 < sum((nr, nc) in shape for nr, nc in neighbors) < 4:
            curves += 1
    return curves

def measure_irregularity(shape: List[Tuple[int, int]]) -> float:
    # Measure irregularity by comparing perimeter to area
    perimeter = calculate_perimeter(shape)
    area = len(shape)
    return perimeter / (area ** 0.5)  # Higher value indicates more irregularity

def calculate_perimeter(shape: List[Tuple[int, int]]) -> int:
    perimeter = 0
    for r, c in shape:
        neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
        perimeter += 4 - sum((nr, nc) in shape for nr, nc in neighbors)
    return perimeter
