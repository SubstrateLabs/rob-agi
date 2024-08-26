from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np

def solve_f4081712(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the f4081712 challenge by analyzing the input grid and generating a condensed
    representation that captures key color relationships and patterns.
    
    The solution involves the following steps:
    1. Analyze the entire input grid, focusing on the central area and key patterns.
    2. Identify the most significant colors and their relationships.
    3. Determine the appropriate output size based on the complexity of the input.
    4. Extract the core pattern from the central area of the input grid.
    5. Generate a smaller output grid that preserves the essence of the input pattern.
    6. Ensure the output maintains key color relationships and relative positions.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: A condensed representation of the input grid, capturing its essential pattern.
    """
    analysis = analyze_grid(input_grid)
    key_colors = identify_key_colors(analysis)
    output_size = determine_output_size(analysis)
    core_pattern = extract_core_pattern(input_grid, analysis)
    output_values = generate_output_grid(core_pattern, key_colors, output_size, analysis)
    return ColoredGrid(values=output_values)

def analyze_grid(grid: ColoredGrid) -> Dict:
    rows, cols = len(grid.values), len(grid.values[0])
    central_area = [row[cols//4:3*cols//4] for row in grid.values[rows//4:3*rows//4]]
    
    analysis = {
        'full_freq': Counter(cell for row in grid.values for cell in row),
        'central_freq': Counter(cell for row in central_area for cell in row),
        'transitions': Counter(),
        'edge_colors': set(grid.values[0] + grid.values[-1] + [row[0] for row in grid.values] + [row[-1] for row in grid.values])
    }
    
    for i in range(rows):
        for j in range(cols):
            if j < cols - 1:
                analysis['transitions'][(grid.values[i][j], grid.values[i][j+1])] += 1
            if i < rows - 1:
                analysis['transitions'][(grid.values[i][j], grid.values[i+1][j])] += 1
    
    return analysis

def identify_key_colors(analysis: Dict) -> List[int]:
    combined_freq = analysis['central_freq'] + Counter({color: count // 2 for color, count in analysis['full_freq'].items()})
    return [color for color, _ in combined_freq.most_common(6)]  # Increased to top 6 colors

def determine_output_size(analysis: Dict) -> Tuple[int, int]:
    unique_colors = len(analysis['central_freq'])
    size = max(4, min(6, unique_colors))  # Adjusted to produce slightly larger outputs
    return (size, size)

def extract_core_pattern(grid: ColoredGrid, analysis: Dict) -> List[List[int]]:
    rows, cols = len(grid.values), len(grid.values[0])
    central_area = [row[cols//4:3*cols//4] for row in grid.values[rows//4:3*rows//4]]
    
    # Convert to numpy array for easier manipulation
    central_array = np.array(central_area)
    
    # Find the most common color in the central area
    most_common_color = max(analysis['central_freq'], key=analysis['central_freq'].get)
    
    # Create a binary mask of the most common color
    mask = (central_array == most_common_color)
    
    # Find the largest contiguous area of the most common color
    labeled_array, num_features = np.zeros_like(mask), 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] and labeled_array[i, j] == 0:
                num_features += 1
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and mask[x, y] and labeled_array[x, y] == 0:
                        labeled_array[x, y] = num_features
                        stack.extend([(x-1, y), (x+1, y), (x, y-1), (x, y+1)])
    
    # Find the largest labeled area
    largest_label = max(range(1, num_features + 1), key=lambda x: np.sum(labeled_array == x))
    largest_area_mask = (labeled_array == largest_label)
    
    # Extract the bounding box of the largest area
    rows, cols = np.where(largest_area_mask)
    top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
    
    # Extract the core pattern
    core_pattern = central_array[top:bottom+1, left:right+1].tolist()
    
    return core_pattern

def generate_output_grid(core_pattern: List[List[int]], key_colors: List[int], output_size: Tuple[int, int], analysis: Dict) -> List[List[int]]:
    rows, cols = output_size
    output = [[0] * cols for _ in range(rows)]
    
    # Scale the core pattern to fit the output size
    scale_factor = min(rows / len(core_pattern), cols / len(core_pattern[0]))
    scaled_pattern = [[core_pattern[int(i/scale_factor)][int(j/scale_factor)] 
                       for j in range(cols)] for i in range(rows)]
    
    # Place the scaled pattern in the output grid
    for r in range(rows):
        for c in range(cols):
            output[r][c] = scaled_pattern[r][c]
    
    # Ensure all key colors are present
    for color in key_colors:
        if color not in [cell for row in output for cell in row]:
            # Find a suitable position to place the missing color
            for r in range(rows):
                for c in range(cols):
                    if output[r][c] not in key_colors:
                        output[r][c] = color
                        break
                if color in [cell for row in output for cell in row]:
                    break
    
    return output
