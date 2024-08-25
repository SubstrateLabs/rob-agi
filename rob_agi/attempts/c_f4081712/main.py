from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple
import numpy as np

def solve_f4081712(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the f4081712 challenge by analyzing the central area of the input grid
    and generating a condensed representation that captures key color relationships.
    
    The solution involves the following steps:
    1. Analyze the central area of the input grid for color frequencies and transitions.
    2. Determine the core color palette based on frequency and significance.
    3. Identify key color relationships and transitions.
    4. Generate an output grid that reflects these relationships.
    5. Refine the output to better match the input's central color distribution.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: A condensed representation of the input grid, focusing on central color relationships.
    """
    central_area = extract_central_area(input_grid)
    color_freq = count_colors(central_area)
    core_palette = determine_core_palette(color_freq)
    color_relationships = identify_color_relationships(central_area)
    output_size = determine_output_size(input_grid, color_freq)
    output_values = generate_output_grid(core_palette, color_relationships, output_size)
    output_values = refine_output(output_values, color_freq, central_area)
    return ColoredGrid(values=output_values)

def extract_central_area(grid: ColoredGrid) -> List[List[int]]:
    rows, cols = len(grid.values), len(grid.values[0])
    start_row, start_col = rows // 4, cols // 4
    end_row, end_col = 3 * rows // 4, 3 * cols // 4
    return [row[start_col:end_col] for row in grid.values[start_row:end_row]]

def count_colors(grid: List[List[int]]) -> Counter:
    return Counter(cell for row in grid for cell in row)

def determine_core_palette(color_freq: Counter) -> List[int]:
    return [color for color, _ in color_freq.most_common(6)]

def identify_color_relationships(grid: List[List[int]]) -> List[Tuple[int, int]]:
    relationships = Counter()
    rows, cols = len(grid), len(grid[0])
    for i in range(rows):
        for j in range(cols):
            if j < cols - 1:
                relationships[(grid[i][j], grid[i][j+1])] += 1
            if i < rows - 1:
                relationships[(grid[i][j], grid[i+1][j])] += 1
    return [pair for pair, _ in relationships.most_common(10)]

def determine_output_size(grid: ColoredGrid, color_freq: Counter) -> Tuple[int, int]:
    unique_colors = len(color_freq)
    input_size = len(grid.values)
    size = max(5, min(8, unique_colors + input_size // 12))
    return (size, size)

def generate_output_grid(core_palette: List[int], color_relationships: List[Tuple[int, int]], output_size: Tuple[int, int]) -> List[List[int]]:
    rows, cols = output_size
    output = [[core_palette[0]] * cols for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            if i == 0 or j == 0 or i == rows-1 or j == cols-1:
                output[i][j] = core_palette[1]
    
    for color1, color2 in color_relationships:
        placed = False
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if output[i][j] == color1 and output[i][j+1] == core_palette[0]:
                    output[i][j+1] = color2
                    placed = True
                    break
            if placed:
                break
    
    return output

def refine_output(output: List[List[int]], color_freq: Counter, central_area: List[List[int]]) -> List[List[int]]:
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
    central_total = sum(color_freq.values())
    for color in input_colors:
        target_count = int((color_freq[color] / central_total) * total_cells)
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
