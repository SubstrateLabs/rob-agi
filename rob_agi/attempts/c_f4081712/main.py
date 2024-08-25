from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple
import numpy as np

def solve_f4081712(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the f4081712 challenge by identifying and extracting significant sub-patterns from the input grid.
    
    The solution involves the following steps:
    1. Analyze the input grid for color frequencies and patterns.
    2. Identify the most significant sub-pattern using a sliding window approach, focusing on the center.
    3. Extract and adjust the chosen sub-pattern, preserving color relationships.
    4. Resize the pattern to match the expected output size (between 5x5 and 8x8).
    5. Ensure the output maintains key features, color diversity, and central patterns from the input.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: A condensed representation of the input grid, focusing on central significant sub-patterns.
    """
    color_freq = count_colors(input_grid)
    sub_pattern = find_significant_subpattern(input_grid)
    output_size = determine_output_size(input_grid, color_freq)
    output_values = resize_pattern(sub_pattern, output_size)
    output_values = fine_tune_output(output_values, color_freq, input_grid)
    return ColoredGrid(values=output_values)

def count_colors(grid: ColoredGrid) -> Counter:
    return Counter(cell for row in grid.values for cell in row)

def determine_output_size(grid: ColoredGrid, color_freq: Counter) -> Tuple[int, int]:
    unique_colors = len(color_freq)
    input_size = len(grid.values)
    size = max(5, min(8, unique_colors + input_size // 12))
    return (size, size)

def find_significant_subpattern(grid: ColoredGrid) -> List[List[int]]:
    rows, cols = len(grid.values), len(grid.values[0])
    center_row, center_col = rows // 2, cols // 2
    max_score = float('-inf')
    best_pattern = None
    
    for window_size in range(5, min(rows, cols) + 1):
        half_window = window_size // 2
        start_row = max(0, center_row - half_window)
        start_col = max(0, center_col - half_window)
        end_row = min(rows, center_row + half_window + 1)
        end_col = min(cols, center_col + half_window + 1)
        
        window = [row[start_col:end_col] for row in grid.values[start_row:end_row]]
        score = calculate_significance(window, grid)
        if score > max_score:
            max_score = score
            best_pattern = window
    
    return best_pattern

def calculate_significance(window: List[List[int]], grid: ColoredGrid) -> float:
    color_variety = len(set(cell for row in window for cell in row))
    color_transitions = sum(window[i][j] != window[i][j+1] for i in range(len(window)) for j in range(len(window[0])-1))
    color_transitions += sum(window[i][j] != window[i+1][j] for i in range(len(window)-1) for j in range(len(window[0])))
    center_weight = 1 + (len(grid.values) // 2 - abs(len(window) // 2 - len(grid.values) // 2)) / len(grid.values)
    return (color_variety * 2 + color_transitions * 0.5) * center_weight

def resize_pattern(pattern: List[List[int]], output_size: Tuple[int, int]) -> List[List[int]]:
    pattern_array = np.array(pattern)
    target_rows, target_cols = output_size
    row_scale = target_rows / len(pattern)
    col_scale = target_cols / len(pattern[0])
    resized = np.zeros(output_size, dtype=int)
    for i in range(target_rows):
        for j in range(target_cols):
            resized[i, j] = pattern[int(i / row_scale)][int(j / col_scale)]
    return resized.tolist()

def fine_tune_output(output: List[List[int]], color_freq: Counter, input_grid: ColoredGrid) -> List[List[int]]:
    output_freq = Counter(cell for row in output for cell in row)
    input_colors = set(color_freq.keys())
    output_colors = set(output_freq.keys())
    
    # Ensure all colors from input are represented in output
    for color in input_colors - output_colors:
        least_common = output_freq.most_common()[-1][0]
        for i in range(len(output)):
            for j in range(len(output[0])):
                if output[i][j] == least_common:
                    output[i][j] = color
                    output_freq[color] += 1
                    output_freq[least_common] -= 1
                    if output_freq[least_common] == 0:
                        del output_freq[least_common]
                    break
            if color in output_freq:
                break
    
    # Adjust color frequencies to better match input
    total_cells = len(output) * len(output[0])
    for color in input_colors:
        target_count = int((color_freq[color] / sum(color_freq.values())) * total_cells)
        current_count = output_freq[color]
        while current_count < target_count:
            for i in range(len(output)):
                for j in range(len(output[0])):
                    if output[i][j] != color and output_freq[output[i][j]] > 1:
                        output[i][j] = color
                        output_freq[color] += 1
                        current_count += 1
                        if current_count == target_count:
                            break
                if current_count == target_count:
                    break
            if current_count == target_count:
                break
    
    return output
