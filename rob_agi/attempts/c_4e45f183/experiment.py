from collections import Counter
from typing import List, Tuple

def analyze_section(section: List[List[int]]) -> Tuple[int, int]:
    colors = [color for row in section for color in row if color != 0]
    if not colors:
        return 0, 0
    color_counts = Counter(colors)
    sorted_colors = sorted(color_counts.items(), key=lambda x: (-x[1], -x[0]))
    return sorted_colors[0][0], sorted_colors[1][0] if len(sorted_colors) > 1 else sorted_colors[0][0]

def analyze_grid(grid: List[List[int]]):
    sections = [
        [grid[i][j:j+5] for i in range(1, 6)] for j in range(1, 18, 6)
    ] + [
        [grid[i][j:j+5] for i in range(7, 12)] for j in range(1, 18, 6)
    ] + [
        [grid[i][j:j+5] for i in range(13, 18)] for j in range(1, 18, 6)
    ]
    
    for i, section in enumerate(sections):
        primary, secondary = analyze_section(section)
        print(f"Section {i+1}: Primary color = {primary}, Secondary color = {secondary}")

# Example input grid
input_grid = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 1, 0],
    [0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 1, 0],
    [0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 1, 0],
    [0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 1, 1, 0, 2, 2, 2, 2, 1, 0],
    [0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 1, 1, 0, 2, 2, 2, 2, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 2, 2, 0, 1, 1, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0],
    [0, 2, 2, 2, 2, 2, 0, 1, 1, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0],
    [0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0],
    [0, 1, 1, 2, 2, 2, 0, 1, 1, 2, 1, 1, 0, 2, 2, 2, 2, 2, 0],
    [0, 1, 1, 2, 2, 2, 0, 1, 1, 2, 1, 1, 0, 2, 2, 2, 2, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0],
    [0, 1, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0],
    [0, 1, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0],
    [0, 1, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0],
    [0, 1, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

analyze_grid(input_grid)
