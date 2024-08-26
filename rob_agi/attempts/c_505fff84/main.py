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
    6. Adjusts the pattern to ensure it starts with a non-red square if possible
    7. Returns the final pattern as a new ColoredGrid
    """
    binary_grid = convert_to_binary(input_grid)
    features = extract_features(binary_grid)
    output_size = determine_output_size(binary_grid, features)
    abstract_pattern = create_abstract_pattern(binary_grid, features, output_size)
    final_pattern = refine_and_balance_pattern(abstract_pattern, features)
    final_pattern = adjust_pattern_start(final_pattern)
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
    elif input_rows <= 5 and input_cols <= 6:
        return (input_rows, input_cols)
    else:
        rows = min(3, input_rows)
        cols = min(6, input_cols)
        return (rows, cols)

def create_abstract_pattern(binary_grid: np.ndarray, features: dict, output_size: Tuple[int, int]) -> np.ndarray:
    output = np.zeros(output_size, dtype=int)
    input_rows, input_cols = binary_grid.shape
    output_rows, output_cols = output_size
    
    # Map significant features
    for i in range(output_rows):
        for j in range(output_cols):
            input_i = int(i * input_rows / output_rows)
            input_j = int(j * input_cols / output_cols)
            region = binary_grid[input_i:input_i+input_rows//output_rows, input_j:input_j+input_cols//output_cols]
            output[i, j] = 1 if np.mean(region) > 0.2 else 0
    
    # Represent frame if detected
    if features['frame']:
        output[0, :] = 1
        output[-1, :] = 1
        output[:, 0] = 1
        output[:, -1] = 1
    
    # Maintain symmetry if detected
    vertical_sym, horizontal_sym = features['symmetry']
    if vertical_sym:
        output = (output + np.fliplr(output)) // 2
    if horizontal_sym:
        output = (output + np.flipud(output)) // 2
    
    # Ensure at least one red square
    if np.sum(output) == 0:
        output[0, -1] = 1  # Place a red square at the top-right corner
    
    return output

def refine_and_balance_pattern(pattern: np.ndarray, features: dict) -> List[List[int]]:
    target_density = features['density']
    current_density = np.mean(pattern)
    total_cells = pattern.size
    target_red_cells = int(round(target_density * total_cells))
    current_red_cells = np.sum(pattern)
    
    while current_red_cells != target_red_cells:
        if current_red_cells < target_red_cells:
            # Add a red square
            zero_indices = np.where(pattern == 0)
            if len(zero_indices[0]) > 0:
                idx = np.random.randint(len(zero_indices[0]))
                pattern[zero_indices[0][idx], zero_indices[1][idx]] = 1
                current_red_cells += 1
        else:
            # Remove a red square
            one_indices = np.where(pattern == 1)
            if len(one_indices[0]) > 0:
                idx = np.random.randint(len(one_indices[0]))
                pattern[one_indices[0][idx], one_indices[1][idx]] = 0
                current_red_cells -= 1
    
    # Ensure at least one red square
    if np.sum(pattern) == 0:
        pattern[-1, -1] = 1  # Place a red square at the bottom-right corner
    
    return [[2 if cell else 0 for cell in row] for row in pattern]
