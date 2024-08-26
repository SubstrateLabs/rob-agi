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
    5. Refines and balances the pattern to match input density and structure
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

def determine_output_size(binary_grid: np.ndarray, features: dict) -> Tuple[int, int]:
    input_rows, input_cols = binary_grid.shape
    aspect_ratio = input_cols / input_rows
    
    if aspect_ratio > 2:
        return (1, 7)
    elif aspect_ratio < 0.5:
        return (5, 1)
    elif input_rows <= 5 and input_cols <= 5:
        return (input_rows, input_cols)
    else:
        return (3, 4) if aspect_ratio > 1 else (4, 3)

def create_abstract_pattern(binary_grid: np.ndarray, features: dict, output_size: Tuple[int, int]) -> np.ndarray:
    output = np.zeros(output_size, dtype=int)
    
    # Map largest component
    if features['largest_component']:
        component = np.array(features['largest_component'])
        min_i, min_j = np.min(component, axis=0)
        max_i, max_j = np.max(component, axis=0)
        component_height, component_width = max_i - min_i + 1, max_j - min_j + 1
        scale_i = output_size[0] / component_height
        scale_j = output_size[1] / component_width
        for i, j in component:
            new_i = int((i - min_i) * scale_i)
            new_j = int((j - min_j) * scale_j)
            if 0 <= new_i < output_size[0] and 0 <= new_j < output_size[1]:
                output[new_i, new_j] = 1
    
    # Represent frame if detected
    if features['frame']:
        output[0, :] = 1
        output[-1, :] = 1
        output[:, 0] = 1
        output[:, -1] = 1
    
    return output

def refine_and_balance_pattern(pattern: np.ndarray, features: dict) -> List[List[int]]:
    target_density = features['density']
    current_density = np.mean(pattern)
    
    while abs(current_density - target_density) > 0.1:
        if current_density < target_density:
            # Add a red square
            zero_indices = np.where(pattern == 0)
            if len(zero_indices[0]) > 0:
                idx = np.random.randint(len(zero_indices[0]))
                pattern[zero_indices[0][idx], zero_indices[1][idx]] = 1
        else:
            # Remove a red square
            one_indices = np.where(pattern == 1)
            if len(one_indices[0]) > 0:
                idx = np.random.randint(len(one_indices[0]))
                pattern[one_indices[0][idx], one_indices[1][idx]] = 0
        
        current_density = np.mean(pattern)
    
    # Ensure at least one red square
    if np.sum(pattern) == 0:
        pattern[0, 0] = 1
    
    return [[2 if cell else 0 for cell in row] for row in pattern]
