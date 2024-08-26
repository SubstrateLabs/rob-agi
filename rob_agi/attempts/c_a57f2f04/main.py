from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_a57f2f04(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a checkerboard pattern to distinct regions.
    
    The function identifies non-sky blue regions in the input grid and replaces them
    with a checkerboard pattern. It uses the first non-black, non-sky blue color found
    in each region for the checkerboard. The pattern is 2x2 for all colors except green (3),
    which uses a 3x3 pattern. The sky blue (8) background remains unchanged.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed output grid.
    """
    output_grid = input_grid.deep_copy()
    regions = find_regions(input_grid)
    
    for top, left, height, width in regions:
        color = determine_color(input_grid, top, left, height, width)
        if color is not None:
            size = 3 if color == 3 else 2
            pattern = generate_checkerboard(color, size, height, width)
            
            for i in range(height):
                for j in range(width):
                    output_grid.values[top + i][left + j] = pattern[i][j]
    
    return output_grid

def find_regions(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    """
    Identifies distinct rectangular regions in the grid that are not sky blue.
    
    Args:
    grid (ColoredGrid): The input grid to analyze.
    
    Returns:
    List[Tuple[int, int, int, int]]: List of (top, left, height, width) for each region.
    """
    regions = []
    rows, cols = grid.get_dimensions()
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and grid.values[i][j] != 8:
                top, left = i, j
                bottom, right = i, j
                
                # Find bottom-right corner of the region
                while bottom + 1 < rows and grid.values[bottom + 1][j] != 8:
                    bottom += 1
                while right + 1 < cols and grid.values[i][right + 1] != 8:
                    right += 1
                
                # Mark region as visited
                for r in range(top, bottom + 1):
                    for c in range(left, right + 1):
                        visited[r][c] = True
                
                regions.append((top, left, bottom - top + 1, right - left + 1))
    
    return regions

def determine_color(grid: ColoredGrid, top: int, left: int, height: int, width: int) -> int:
    """
    Determines the color for a region by finding the first non-black, non-sky blue color.
    
    Args:
    grid (ColoredGrid): The input grid.
    top, left, height, width: The region's coordinates and dimensions.
    
    Returns:
    int: The determined color for the region, or None if no suitable color is found.
    """
    for r in range(top, top + height):
        for c in range(left, left + width):
            color = grid.values[r][c]
            if color not in {0, 8}:
                return color
    return None

def generate_checkerboard(color: int, size: int, height: int, width: int) -> List[List[int]]:
    """
    Generates a checkerboard pattern of the specified color and size.
    
    Args:
    color (int): The color to use for the checkerboard.
    size (int): The size of each square in the checkerboard (2 or 3).
    height (int): The height of the region.
    width (int): The width of the region.
    
    Returns:
    List[List[int]]: 2D list representing the checkerboard pattern.
    """
    pattern = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            if ((i // size) + (j // size)) % 2 == 0:
                pattern[i][j] = color
    return pattern
