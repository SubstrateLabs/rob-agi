from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import numpy as np

def solve_505fff84(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts the most significant pattern of red squares from the input grid.
    
    The function performs the following steps:
    1. Analyzes the input grid to determine its characteristics (size, density, patterns)
    2. Determines the appropriate output size based on input characteristics
    3. Creates an abstract pattern that captures the essence of the input
    4. Refines the pattern to match input structure, focusing on significant features
    5. Adjusts the pattern to maintain proper density and structure
    6. Returns the final pattern as a new ColoredGrid
    """
    binary_grid = convert_to_binary(input_grid)
    features = extract_features(binary_grid)
    output_size = determine_output_size(binary_grid, features)
    abstract_pattern = create_abstract_pattern(binary_grid, features, output_size)
    final_pattern = refine_and_balance_pattern(abstract_pattern, features)
    return ColoredGrid(values=final_pattern)

def convert_to_binary(grid: ColoredGrid) -> np.ndarray:
    return np.array(grid.values) == 2

def extract_features(binary_grid: np.ndarray) -> dict:
    features = {}
    features['density'] = np.mean(binary_grid)
    features['row_density'] = np.mean(binary_grid, axis=1)
    features['col_density'] = np.mean(binary_grid, axis=0)
    features['largest_component'] = largest_connected_component(binary_grid)
    features['frame'] = detect_frame(binary_grid)
    features['symmetry'] = detect_symmetry(binary_grid)
    return features

def largest_connected_component(binary_grid: np.ndarray) -> List[Tuple[int, int]]:
    def dfs(i, j, component):
        if i < 0 or i >= binary_grid.shape[0] or j < 0 or j >= binary_grid.shape[1] or not binary_grid[i, j] or (i, j) in visited:
            return
        visited.add((i, j))
        component.append((i, j))
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            dfs(i + di, j + dj, component)

    visited = set()
    largest_component = []
    for i in range(binary_grid.shape[0]):
        for j in range(binary_grid.shape[1]):
            if binary_grid[i, j] and (i, j) not in visited:
                component = []
                dfs(i, j, component)
                if len(component) > len(largest_component):
                    largest_component = component
    return largest_component

def detect_frame(binary_grid: np.ndarray) -> bool:
    rows, cols = binary_grid.shape
    top = np.any(binary_grid[0, :])
    bottom = np.any(binary_grid[-1, :])
    left = np.any(binary_grid[:, 0])
    right = np.any(binary_grid[:, -1])
    return (top and bottom and left and right)

def detect_symmetry(binary_grid: np.ndarray) -> Tuple[bool, bool]:
    vertical_symmetry = np.all(binary_grid == np.fliplr(binary_grid))
    horizontal_symmetry = np.all(binary_grid == np.flipud(binary_grid))
    return vertical_symmetry, horizontal_symmetry

def determine_output_size(binary_grid: np.ndarray, features: dict) -> Tuple[int, int]:
    input_rows, input_cols = binary_grid.shape
    aspect_ratio = input_cols / input_rows
    red_squares = np.sum(binary_grid)
    
    if aspect_ratio > 2:
        return (1, min(7, input_cols))
    elif aspect_ratio < 0.5:
        return (min(5, input_rows), 1)
    elif red_squares < 10:
        return (3, 3)
    elif red_squares < 20:
        return (4, 4)
    else:
        return (5, 5)

def create_abstract_pattern(binary_grid: np.ndarray, features: dict, output_size: Tuple[int, int]) -> np.ndarray:
    output = np.zeros(output_size, dtype=int)
    input_rows, input_cols = binary_grid.shape
    output_rows, output_cols = output_size
    
    # Create frame if input has frame-like structure
    if features['frame']:
        output[0, :] = 1
        output[-1, :] = 1
        output[:, 0] = 1
        output[:, -1] = 1
    
    # Represent largest component
    largest_comp = features['largest_component']
    if largest_comp:
        comp_rows = [r for r, _ in largest_comp]
        comp_cols = [c for _, c in largest_comp]
        min_row, max_row = min(comp_rows), max(comp_rows)
        min_col, max_col = min(comp_cols), max(comp_cols)
        
        # Map component to output grid
        out_min_row = int(min_row / input_rows * output_rows)
        out_max_row = int(max_row / input_rows * output_rows)
        out_min_col = int(min_col / input_cols * output_cols)
        out_max_col = int(max_col / input_cols * output_cols)
        
        output[out_min_row:out_max_row+1, out_min_col:out_max_col+1] = 1
    
    return output

def refine_and_balance_pattern(pattern: np.ndarray, features: dict) -> List[List[int]]:
    target_density = features['density']
    current_density = np.mean(pattern)
    
    # Adjust density
    while abs(current_density - target_density) > 0.1:
        if current_density > target_density:
            # Remove a random red square
            red_squares = np.where(pattern == 1)
            if len(red_squares[0]) > 0:
                idx = np.random.randint(len(red_squares[0]))
                pattern[red_squares[0][idx], red_squares[1][idx]] = 0
        else:
            # Add a random black square
            black_squares = np.where(pattern == 0)
            if len(black_squares[0]) > 0:
                idx = np.random.randint(len(black_squares[0]))
                pattern[black_squares[0][idx], black_squares[1][idx]] = 1
        current_density = np.mean(pattern)
    
    # Ensure at least one red square
    if np.sum(pattern) == 0:
        pattern[0, 0] = 1
    
    # Convert to list of lists and map 1 to 2 (red)
    return [[2 if cell else 0 for cell in row] for row in pattern]
