from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np

def solve_f4081712(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the f4081712 challenge by analyzing the input grid and generating a condensed
    representation that captures key color relationships, patterns, and structural elements.
    
    The solution involves the following steps:
    1. Analyze the entire input grid, detecting symmetry, regions, and patterns.
    2. Identify key structural elements and color relationships.
    3. Create an abstract representation of the input grid.
    4. Determine the appropriate output size and structure.
    5. Generate a smaller output grid that preserves the essence of the input pattern.
    6. Refine the output to ensure it captures the most important features of the input.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: A condensed representation of the input grid, capturing its essential pattern and structure.
    """
    analysis = analyze_grid(input_grid)
    abstract_rep = create_abstract_representation(input_grid, analysis)
    output_size = determine_output_size(analysis)
    output_values = generate_output_grid(abstract_rep, output_size, analysis)
    return ColoredGrid(values=output_values)

def analyze_grid(grid: ColoredGrid) -> Dict:
    rows, cols = len(grid.values), len(grid.values[0])
    central_area = [row[cols//4:3*cols//4] for row in grid.values[rows//4:3*rows//4]]
    
    analysis = {
        'full_freq': Counter(cell for row in grid.values for cell in row),
        'central_freq': Counter(cell for row in central_area for cell in row),
        'transitions': Counter(),
        'edge_colors': set(grid.values[0] + grid.values[-1] + [row[0] for row in grid.values] + [row[-1] for row in grid.values]),
        'symmetry': detect_symmetry(grid),
        'regions': detect_regions(grid),
    }
    
    for i in range(rows):
        for j in range(cols):
            if j < cols - 1:
                analysis['transitions'][(grid.values[i][j], grid.values[i][j+1])] += 1
            if i < rows - 1:
                analysis['transitions'][(grid.values[i][j], grid.values[i+1][j])] += 1
    
    return analysis

def detect_symmetry(grid: ColoredGrid) -> Dict[str, bool]:
    rows, cols = len(grid.values), len(grid.values[0])
    horizontal = all(grid.values[i] == grid.values[rows-1-i] for i in range(rows//2))
    vertical = all(grid.values[i][j] == grid.values[i][cols-1-j] for i in range(rows) for j in range(cols//2))
    return {'horizontal': horizontal, 'vertical': vertical}

def detect_regions(grid: ColoredGrid) -> List[Dict]:
    rows, cols = len(grid.values), len(grid.values[0])
    visited = set()
    regions = []
    
    def dfs(r, c, color):
        if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols or grid.values[r][c] != color:
            return []
        visited.add((r, c))
        region = [(r, c)]
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            region.extend(dfs(r+dr, c+dc, color))
        return region
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited:
                region = dfs(r, c, grid.values[r][c])
                if region:
                    regions.append({'color': grid.values[r][c], 'cells': region})
    
    return regions

def create_abstract_representation(grid: ColoredGrid, analysis: Dict) -> Dict:
    abstract_rep = {
        'key_colors': [color for color, _ in analysis['full_freq'].most_common(5)],
        'central_pattern': extract_central_pattern(grid),
        'edge_pattern': extract_edge_pattern(grid),
        'symmetry': analysis['symmetry'],
        'major_regions': sorted(analysis['regions'], key=lambda r: len(r['cells']), reverse=True)[:3]
    }
    return abstract_rep

def extract_central_pattern(grid: ColoredGrid) -> List[List[int]]:
    rows, cols = len(grid.values), len(grid.values[0])
    return [row[cols//4:3*cols//4] for row in grid.values[rows//4:3*rows//4]]

def extract_edge_pattern(grid: ColoredGrid) -> List[int]:
    rows, cols = len(grid.values), len(grid.values[0])
    return grid.values[0] + [row[-1] for row in grid.values[1:-1]] + grid.values[-1][::-1] + [row[0] for row in grid.values[-2:0:-1]]

def determine_output_size(analysis: Dict) -> Tuple[int, int]:
    unique_colors = len(analysis['central_freq'])
    size = max(5, min(7, unique_colors))
    return (size, size)

def generate_output_grid(abstract_rep: Dict, output_size: Tuple[int, int], analysis: Dict) -> List[List[int]]:
    rows, cols = output_size
    output = [[0] * cols for _ in range(rows)]
    
    # Place central pattern
    central_pattern = abstract_rep['central_pattern']
    scale_factor = min(rows / len(central_pattern), cols / len(central_pattern[0]))
    for r in range(rows):
        for c in range(cols):
            output[r][c] = central_pattern[int(r/scale_factor)][int(c/scale_factor)]
    
    # Add edge colors
    edge_colors = abstract_rep['edge_pattern']
    for i in range(rows):
        output[i][0] = edge_colors[i % len(edge_colors)]
        output[i][-1] = edge_colors[(i + len(edge_colors)//2) % len(edge_colors)]
    for j in range(cols):
        output[0][j] = edge_colors[(j + len(edge_colors)//4) % len(edge_colors)]
        output[-1][j] = edge_colors[(j + 3*len(edge_colors)//4) % len(edge_colors)]
    
    # Ensure key colors are present
    for color in abstract_rep['key_colors']:
        if color not in [cell for row in output for cell in row]:
            r, c = rows//2, cols//2
            while output[r][c] in abstract_rep['key_colors']:
                r = (r + 1) % rows
                c = (c + 1) % cols
            output[r][c] = color
    
    # Apply symmetry if detected
    if abstract_rep['symmetry']['horizontal']:
        for r in range(rows//2):
            output[rows-1-r] = output[r].copy()
    if abstract_rep['symmetry']['vertical']:
        for r in range(rows):
            for c in range(cols//2):
                output[r][cols-1-c] = output[r][c]
    
    return output
