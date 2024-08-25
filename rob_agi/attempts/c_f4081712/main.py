from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple, Dict
import random

def solve_f4081712(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the f4081712 challenge by analyzing the input grid and generating a condensed
    representation that captures key color relationships and patterns.
    
    The solution involves the following steps:
    1. Analyze the entire input grid, with focus on the central area.
    2. Identify key colors based on frequency and significance in forming patterns.
    3. Determine color relationships and transitions.
    4. Generate an output grid that reflects these relationships and the essence of the input pattern.
    5. Refine the output to balance color distribution and capture diagonal patterns if present.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: A condensed representation of the input grid, capturing its essential pattern.
    """
    grid_analysis = analyze_grid(input_grid)
    key_colors = identify_key_colors(grid_analysis)
    color_relationships = determine_color_relationships(grid_analysis, key_colors)
    output_size = determine_output_size(grid_analysis)
    output_values = generate_output_grid(key_colors, color_relationships, output_size)
    output_values = refine_output(output_values, grid_analysis)
    return ColoredGrid(values=output_values)

def analyze_grid(grid: ColoredGrid) -> Dict:
    rows, cols = len(grid.values), len(grid.values[0])
    central_area = [row[cols//4:3*cols//4] for row in grid.values[rows//4:3*rows//4]]
    
    analysis = {
        'full_freq': Counter(cell for row in grid.values for cell in row),
        'central_freq': Counter(cell for row in central_area for cell in row),
        'transitions': Counter(),
        'diagonals': Counter()
    }
    
    for i in range(rows):
        for j in range(cols):
            if j < cols - 1:
                analysis['transitions'][(grid.values[i][j], grid.values[i][j+1])] += 1
            if i < rows - 1:
                analysis['transitions'][(grid.values[i][j], grid.values[i+1][j])] += 1
            if i < rows - 1 and j < cols - 1:
                analysis['diagonals'][(grid.values[i][j], grid.values[i+1][j+1])] += 1
    
    return analysis

def identify_key_colors(analysis: Dict) -> List[int]:
    combined_freq = analysis['full_freq'] + analysis['central_freq']
    return [color for color, _ in combined_freq.most_common(6)]

def determine_color_relationships(analysis: Dict, key_colors: List[int]) -> List[Tuple[int, int]]:
    relationships = analysis['transitions'] + analysis['diagonals']
    return [pair for pair, _ in relationships.most_common(10) if pair[0] in key_colors and pair[1] in key_colors]

def determine_output_size(analysis: Dict) -> Tuple[int, int]:
    unique_colors = len(analysis['central_freq'])
    size = max(3, min(8, unique_colors + 2))
    return (size, size)

def generate_output_grid(key_colors: List[int], color_relationships: List[Tuple[int, int]], output_size: Tuple[int, int]) -> List[List[int]]:
    rows, cols = output_size
    output = [[key_colors[0]] * cols for _ in range(rows)]
    
    # Place second most common color
    for i in range(rows):
        output[i][0] = output[i][-1] = key_colors[1]
    for j in range(cols):
        output[0][j] = output[-1][j] = key_colors[1]
    
    # Place other colors based on relationships
    for color1, color2 in color_relationships:
        for _ in range(2):  # Try to place each relationship twice
            i, j = random.randint(1, rows-2), random.randint(1, cols-2)
            if output[i][j] == key_colors[0]:
                output[i][j] = color1
                if j < cols-2 and output[i][j+1] == key_colors[0]:
                    output[i][j+1] = color2
                elif i < rows-2 and output[i+1][j] == key_colors[0]:
                    output[i+1][j] = color2
    
    return output

def refine_output(output: List[List[int]], analysis: Dict) -> List[List[int]]:
    rows, cols = len(output), len(output[0])
    target_freq = analysis['central_freq']
    current_freq = Counter(cell for row in output for cell in row)
    
    # Ensure all key colors are present
    for color in target_freq:
        if color not in current_freq:
            i, j = random.randint(0, rows-1), random.randint(0, cols-1)
            output[i][j] = color
            current_freq[color] += 1
    
    # Adjust frequencies
    total_cells = rows * cols
    for color, count in target_freq.items():
        target = int((count / sum(target_freq.values())) * total_cells)
        while current_freq[color] < target:
            i, j = random.randint(0, rows-1), random.randint(0, cols-1)
            if output[i][j] != color and current_freq[output[i][j]] > 1:
                current_freq[output[i][j]] -= 1
                output[i][j] = color
                current_freq[color] += 1
    
    # Add diagonal patterns if present in input
    if any(count > len(analysis['transitions']) / 10 for count in analysis['diagonals'].values()):
        for i in range(min(rows, cols) - 1):
            output[i][i] = output[i+1][i+1]
    
    return output
