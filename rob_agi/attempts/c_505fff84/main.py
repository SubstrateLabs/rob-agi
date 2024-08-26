from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import numpy as np

def solve_505fff84(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts the most significant pattern of red squares from the input grid.
    
    The function performs the following steps:
    1. Converts the input grid to a binary representation (1 for red, 0 for others)
    2. Analyzes the distribution and structure of red squares
    3. Determines the appropriate output size based on input characteristics
    4. Creates an abstract pattern that captures the essence of the input
    5. Refines the pattern to match input structure, focusing on frame-like patterns
    6. Balances the pattern to match input density
    7. Returns the final pattern as a new ColoredGrid
    """
    binary_grid = convert_to_binary(input_grid)
    features = extract_features(binary_grid)
    output_size = determine_output_size(binary_grid, features)
    abstract_pattern = create_frame_pattern(binary_grid, features, output_size)
    final_pattern = refine_and_balance_pattern(abstract_pattern, features)
    return ColoredGrid(values=final_pattern)

def adjust_pattern_start(pattern: List[List[int]]) -> List[List[int]]:
    """Ensures the pattern starts with a non-red square if possible."""
    if len(pattern) == 1 and len(pattern[0]) > 1 and pattern[0][0] == 2:
        # For 1D patterns, shift the pattern right if it starts with red
        return [[0] + pattern[0][:-1]]
    return pattern

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
    
    if aspect_ratio > 2:
        return (1, min(7, input_cols))
    elif aspect_ratio < 0.5:
        return (min(5, input_rows), 1)
    else:
        return (5, 5)  # Use a fixed 5x5 grid for most cases

def create_frame_pattern(binary_grid: np.ndarray, features: dict, output_size: Tuple[int, int]) -> np.ndarray:
    output = np.zeros(output_size, dtype=int)
    input_rows, input_cols = binary_grid.shape
    output_rows, output_cols = output_size
    
    # Create frame
    output[0, :] = 1
    output[-1, :] = 1
    output[:, 0] = 1
    output[:, -1] = 1
    
    # Fill interior based on input density
    interior_density = np.mean(binary_grid[1:-1, 1:-1])
    interior_threshold = 0.3
    if interior_density > interior_threshold:
        output[1:-1, 1:-1] = 1
    
    # Adjust corners based on input
    if not np.any(binary_grid[0:input_rows//3, 0:input_cols//3]):
        output[0, 0] = 0
    if not np.any(binary_grid[0:input_rows//3, -input_cols//3:]):
        output[0, -1] = 0
    if not np.any(binary_grid[-input_rows//3:, 0:input_cols//3]):
        output[-1, 0] = 0
    if not np.any(binary_grid[-input_rows//3:, -input_cols//3:]):
        output[-1, -1] = 0
    
    return output

def refine_and_balance_pattern(pattern: np.ndarray, features: dict) -> List[List[int]]:
    target_density = features['density']
    current_density = np.mean(pattern)
    
    # Adjust density
    if current_density > target_density:
        # Remove red squares from the interior
        interior = pattern[1:-1, 1:-1]
        interior_ones = np.where(interior == 1)
        num_to_remove = int((current_density - target_density) * pattern.size)
        for _ in range(min(num_to_remove, len(interior_ones[0]))):
            idx = np.random.randint(len(interior_ones[0]))
            pattern[1 + interior_ones[0][idx], 1 + interior_ones[1][idx]] = 0
    elif current_density < target_density:
        # Add red squares to the interior
        interior = pattern[1:-1, 1:-1]
        interior_zeros = np.where(interior == 0)
        num_to_add = int((target_density - current_density) * pattern.size)
        for _ in range(min(num_to_add, len(interior_zeros[0]))):
            idx = np.random.randint(len(interior_zeros[0]))
            pattern[1 + interior_zeros[0][idx], 1 + interior_zeros[1][idx]] = 1
    
    # Ensure at least one red square
    if np.sum(pattern) == 0:
        pattern[0, 0] = 1  # Place a red square at the top-left corner
    
    return [[2 if cell else 0 for cell in row] for row in pattern]
