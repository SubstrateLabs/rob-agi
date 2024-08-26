from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e9c9d9a1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling rectangles formed by green (3) lines.
    
    The solution:
    1. Identifies all horizontal and vertical green (3) lines.
    2. Creates a conceptual grid of rectangles.
    3. Fills these rectangles based on their position:
       - Top-left rectangle: red (2)
       - Top-right rectangle: yellow (4)
       - Bottom-left rectangle: blue (1)
       - Bottom-right rectangle: sky blue (8)
       - Middle rectangles: orange (7), except for leftmost and rightmost columns which remain black (0)
    4. Preserves the green (3) lines and any non-black (0) cells from the input.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid according to the pattern.
    """
    def find_lines(grid: ColoredGrid) -> Tuple[List[int], List[int]]:
        h_lines = [i for i, row in enumerate(grid.values) if all(cell == 3 for cell in row)]
        v_lines = [j for j in range(len(grid.values[0])) if all(row[j] == 3 for row in grid.values)]
        return h_lines, v_lines
    
    def fill_rectangle(grid: ColoredGrid, top: int, left: int, bottom: int, right: int, color: int) -> None:
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if grid.values[r][c] == 0:
                    grid.values[r][c] = color
    
    # Create a copy of the input grid
    output_grid = input_grid.deep_copy()
    
    # Find horizontal and vertical green lines
    h_lines, v_lines = find_lines(input_grid)
    
    # Add grid boundaries if not present
    if 0 not in h_lines:
        h_lines.insert(0, 0)
    if len(output_grid.values) - 1 not in h_lines:
        h_lines.append(len(output_grid.values) - 1)
    if 0 not in v_lines:
        v_lines.insert(0, 0)
    if len(output_grid.values[0]) - 1 not in v_lines:
        v_lines.append(len(output_grid.values[0]) - 1)
    
    # Fill rectangles based on their position
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            top, bottom = h_lines[i], h_lines[i + 1]
            left, right = v_lines[j], v_lines[j + 1]
            
            if i == 0 and j == 0:
                color = 2  # Red for top-left
            elif i == 0 and j == len(v_lines) - 2:
                color = 4  # Yellow for top-right
            elif i == len(h_lines) - 2 and j == 0:
                color = 1  # Blue for bottom-left
            elif i == len(h_lines) - 2 and j == len(v_lines) - 2:
                color = 8  # Sky blue for bottom-right
            elif j == 0 or j == len(v_lines) - 2:
                color = 0  # Black for leftmost and rightmost columns
            else:
                color = 7  # Orange for middle rectangles
            
            fill_rectangle(output_grid, top, left, bottom, right, color)
    
    return output_grid
