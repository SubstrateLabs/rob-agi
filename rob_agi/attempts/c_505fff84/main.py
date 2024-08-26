from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import numpy as np

def solve_505fff84(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts the most significant pattern of red squares from the input grid.
    
    The function performs the following steps:
    1. Converts the input grid to a binary representation (1 for red, 0 for others)
    2. Analyzes the distribution of red squares and identifies key features
    3. Generates candidate patterns based on the most significant features
    4. Evaluates and selects the best candidate pattern
    5. Refines the selected pattern to ensure it captures the essence of the input
    6. Returns the final pattern as a new ColoredGrid
    """
    binary_grid = convert_to_binary(input_grid)
    features = extract_features(binary_grid)
    candidates = generate_candidates(features, binary_grid)
    best_candidate = evaluate_candidates(candidates, binary_grid)
    final_pattern = refine_pattern(best_candidate, binary_grid)
    return final_pattern

def convert_to_binary(grid: ColoredGrid) -> np.ndarray:
    return np.array(grid.values) == 2

def extract_features(binary_grid: np.ndarray) -> dict:
    features = {}
    features['density'] = np.mean(binary_grid)
    features['row_density'] = np.mean(binary_grid, axis=1)
    features['col_density'] = np.mean(binary_grid, axis=0)
    features['largest_component'] = largest_connected_component(binary_grid)
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

def generate_candidates(features: dict, binary_grid: np.ndarray) -> List[np.ndarray]:
    candidates = []
    
    # Candidate 1: Based on overall density
    size = max(2, min(6, int(np.sqrt(np.sum(binary_grid)))))
    candidate = np.random.rand(size, size) < features['density']
    candidates.append(candidate)
    
    # Candidate 2: Based on largest connected component
    if features['largest_component']:
        min_i = min(i for i, j in features['largest_component'])
        max_i = max(i for i, j in features['largest_component'])
        min_j = min(j for i, j in features['largest_component'])
        max_j = max(j for i, j in features['largest_component'])
        candidate = binary_grid[min_i:max_i+1, min_j:max_j+1]
        candidates.append(candidate)
    
    # Candidate 3: Based on row and column densities
    rows = np.argsort(features['row_density'])[-3:]
    cols = np.argsort(features['col_density'])[-3:]
    candidate = binary_grid[np.ix_(rows, cols)]
    candidates.append(candidate)
    
    return candidates

def evaluate_candidates(candidates: List[np.ndarray], binary_grid: np.ndarray) -> np.ndarray:
    best_score = -1
    best_candidate = None
    for candidate in candidates:
        score = evaluate_pattern(candidate, binary_grid)
        if score > best_score:
            best_score = score
            best_candidate = candidate
    return best_candidate

def evaluate_pattern(pattern: np.ndarray, binary_grid: np.ndarray) -> float:
    pattern_density = np.mean(pattern)
    grid_density = np.mean(binary_grid)
    size_score = 1 / (np.abs(np.log(pattern.size / binary_grid.size)) + 1)
    density_score = 1 / (np.abs(pattern_density - grid_density) + 0.1)
    return size_score * density_score

def refine_pattern(pattern: np.ndarray, binary_grid: np.ndarray) -> ColoredGrid:
    # Ensure minimum size
    while pattern.shape[0] < 2 or pattern.shape[1] < 2:
        pattern = np.pad(pattern, ((0, 1), (0, 1)), mode='edge')
    
    # Ensure at least one red square
    if np.sum(pattern) == 0:
        pattern[0, 0] = 1
    
    # Convert back to ColoredGrid format
    values = [[2 if cell else 0 for cell in row] for row in pattern]
    return ColoredGrid(values=values)
