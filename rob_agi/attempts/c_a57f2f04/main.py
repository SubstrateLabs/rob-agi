from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_a57f2f04(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a checkerboard pattern to distinct regions.
    
    The function identifies non-black, non-border colored regions in the input grid
    and replaces them with a checkerboard pattern of the same color. The pattern is
    2x2 for all colors except green (3), which uses a 3x3 pattern. Black regions and
    the sky blue (8) border remain unchanged.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed output grid.
    """
    output_grid = input_grid.deep_copy()
    
    for top, left, height, width in find_regions(input_grid):
        region = input_grid.extract_subgrid(top, left, height, width)
        colors = set(cell for row in region.values for cell in row) - {0, 8}
        
        if colors:
            color = next(iter(colors))  # Choose any non-black, non-border color
            size = 3 if color == 3 else 2  # 3x3 for green (3), 2x2 for others
            pattern = generate_checkerboard(color, size, height, width)
            
            for i in range(height):
                for j in range(width):
                    output_grid.values[top + i][left + j] = pattern[i][j]
    
    return output_grid

def find_regions(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    """
    Identifies distinct rectangular regions in the grid.
    
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
            if (i // size + j // size) % 2 == 0:
                pattern[i][j] = color
    return pattern
