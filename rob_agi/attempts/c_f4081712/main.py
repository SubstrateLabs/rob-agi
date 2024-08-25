from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans

def solve_f4081712(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the f4081712 challenge by identifying and extracting significant sub-patterns from the input grid.
    
    The solution involves the following steps:
    1. Analyze the input grid for color frequencies, patterns, and significant areas.
    2. Identify the most significant sub-pattern using a sliding window approach.
    3. Extract and adjust the chosen sub-pattern.
    4. Resize and fine-tune the pattern to match the expected output size.
    5. Ensure the output maintains key features and color relationships from the input.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: A condensed representation of the input grid, focusing on significant sub-patterns.
    """
    # Step 1: Analyze the input grid
    color_freq = count_colors(input_grid)
    
    # Step 2: Identify significant sub-pattern
    sub_pattern = find_significant_subpattern(input_grid)
    
    # Step 3 & 4: Extract, adjust, and resize the sub-pattern
    output_size = determine_output_size(input_grid, color_freq)
    output_values = resize_pattern(sub_pattern, output_size)
    
    # Step 5: Fine-tune the output
    output_values = fine_tune_output(output_values, color_freq)
    
    return ColoredGrid(values=output_values)

def count_colors(grid: ColoredGrid) -> Counter:
    return Counter(cell for row in grid.values for cell in row)

def determine_output_size(grid: ColoredGrid, color_freq: Counter) -> Tuple[int, int]:
    unique_colors = len(color_freq)
    input_size = len(grid.values)
    
    # Heuristic: output size is between 5x5 and 8x8, scaled by unique colors and input size
    size = max(5, min(8, unique_colors + input_size // 8))
    return (size, size)

def find_significant_subpattern(grid: ColoredGrid) -> List[List[int]]:
    rows, cols = len(grid.values), len(grid.values[0])
    max_score = float('-inf')
    best_pattern = None
    
    for window_size in range(5, 9):
        for i in range(rows - window_size + 1):
            for j in range(cols - window_size + 1):
                window = [row[j:j+window_size] for row in grid.values[i:i+window_size]]
                score = calculate_significance(window, grid)
                if score > max_score:
                    max_score = score
                    best_pattern = window
    
    return best_pattern

def calculate_significance(window: List[List[int]], grid: ColoredGrid) -> float:
    color_variety = len(set(cell for row in window for cell in row))
    green_presence = sum(1 for row in window for cell in row if cell == 3)
    contrast = sum(abs(window[i][j] - window[i+1][j]) + abs(window[i][j] - window[i][j+1])
                   for i in range(len(window)-1) for j in range(len(window)-1))
    
    center_weight = 1 + (len(grid.values) // 2 - abs(len(window) // 2 - len(grid.values) // 2)) / len(grid.values)
    
    return (color_variety + green_presence * 2 + contrast * 0.1) * center_weight

def resize_pattern(pattern: List[List[int]], output_size: Tuple[int, int]) -> List[List[int]]:
    pattern_array = np.array(pattern)
    target_rows, target_cols = output_size
    
    if len(pattern) > target_rows or len(pattern[0]) > target_cols:
        # Downscale using k-means clustering
        flat_pattern = pattern_array.reshape(-1, 1)
        kmeans = KMeans(n_clusters=target_rows * target_cols, random_state=42)
        kmeans.fit(flat_pattern)
        resized = kmeans.cluster_centers_.reshape(target_rows, target_cols).astype(int)
    else:
        # Upscale using nearest neighbor interpolation
        row_scale = target_rows / len(pattern)
        col_scale = target_cols / len(pattern[0])
        resized = np.zeros(output_size, dtype=int)
        for i in range(target_rows):
            for j in range(target_cols):
                resized[i, j] = pattern[int(i / row_scale)][int(j / col_scale)]
    
    return resized.tolist()

def fine_tune_output(output: List[List[int]], color_freq: Counter) -> List[List[int]]:
    flat_output = [cell for row in output for cell in row]
    output_freq = Counter(flat_output)
    
    # Ensure all colors from input are represented in output
    for color in color_freq:
        if color not in output_freq:
            least_common = output_freq.most_common()[-1][0]
            for i in range(len(output)):
                for j in range(len(output[0])):
                    if output[i][j] == least_common:
                        output[i][j] = color
                        break
                if color in output_freq:
                    break
    
    return output
