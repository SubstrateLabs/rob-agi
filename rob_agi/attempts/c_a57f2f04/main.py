from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional
from collections import deque

def solve_a57f2f04(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying specific patterns to distinct regions.
    
    The function identifies non-sky blue regions in the input grid and replaces them
    with specific patterns based on the color and position of non-black elements:
    - For red (2): Alternating columns pattern, starting with the color.
    - For yellow (4): 2x2 checkerboard pattern, starting with the color.
    - For green (3): 3x3 pattern with color in corners and center.
    - For blue (1): 2x2 checkerboard pattern, starting with the color.
    The sky blue (8) background remains unchanged.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed output grid.
    """
    output_grid = input_grid.deep_copy()
    regions = find_regions(input_grid)
    
    for top, left, height, width in regions:
        color, pattern_type = determine_pattern(input_grid, top, left, height, width)
        if color is not None:
            pattern = generate_pattern(color, pattern_type, height, width)
            apply_pattern(output_grid, pattern, top, left, height, width)
    
    return output_grid

def find_regions(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    regions = []
    rows, cols = grid.get_dimensions()
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and grid.values[i][j] != 8:
                top, left = i, j
                bottom, right = i, j
                
                while bottom + 1 < rows and grid.values[bottom + 1][j] != 8:
                    bottom += 1
                while right + 1 < cols and grid.values[i][right + 1] != 8:
                    right += 1
                
                for r in range(top, bottom + 1):
                    for c in range(left, right + 1):
                        visited[r][c] = True
                
                regions.append((top, left, bottom - top + 1, right - left + 1))
    
    return regions

def determine_pattern(grid: ColoredGrid, top: int, left: int, height: int, width: int) -> Tuple[Optional[int], str]:
    color = None
    pattern_type = "checkerboard"
    
    for r in range(top, top + height):
        for c in range(left, left + width):
            cell_color = grid.values[r][c]
            if cell_color not in {0, 8}:
                color = cell_color
                if color == 2:  # Red
                    pattern_type = "alternating_columns"
                elif color == 3:  # Green
                    pattern_type = "3x3"
                break
        if color:
            break
    
    return color, pattern_type

def generate_pattern(color: int, pattern_type: str, height: int, width: int) -> List[List[int]]:
    pattern = [[0 for _ in range(width)] for _ in range(height)]
    
    if pattern_type == "checkerboard":
        for i in range(height):
            for j in range(width):
                if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                    pattern[i][j] = color
    elif pattern_type == "alternating_columns":
        for j in range(width):
            if j % 2 == 0:
                for i in range(height):
                    pattern[i][j] = color
    elif pattern_type == "3x3":
        base_pattern = [[color, 0, color], [0, color, 0], [color, 0, color]]
        for i in range(height):
            for j in range(width):
                pattern[i][j] = base_pattern[i % 3][j % 3]
    
    return pattern

def apply_pattern(grid: ColoredGrid, pattern: List[List[int]], top: int, left: int, height: int, width: int):
    for i in range(height):
        for j in range(width):
            grid.values[top + i][left + j] = pattern[i][j]
