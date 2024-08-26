from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e9c9d9a1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling sections formed by green (3) lines.
    
    The solution:
    1. Identifies the frame formed by the outermost green (3) lines or grid edges.
    2. Fills sections based on their position relative to the frame:
       - Top-left corner: red (2)
       - Top-right corner: yellow (4)
       - Bottom-left corner: blue (1)
       - Bottom-right corner: sky blue (8)
       - Inside the frame: orange (7)
       - On the frame but not in corners: remains black (0)
    3. Preserves all green (3) lines and any non-black (0) cells from the input.
    4. Handles edge cases where there might be few or no green lines.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid according to the pattern.
    """
    def find_frame(grid: ColoredGrid) -> Tuple[int, int, int, int]:
        rows, cols = len(grid.values), len(grid.values[0])
        top = next((i for i, row in enumerate(grid.values) if 3 in row), 0)
        bottom = next((rows - 1 - i for i, row in enumerate(reversed(grid.values)) if 3 in row), rows - 1)
        left = next((j for j in range(cols) if any(row[j] == 3 for row in grid.values)), 0)
        right = next((cols - 1 - j for j in range(cols) if any(row[cols-1-j] == 3 for row in grid.values)), cols - 1)
        return top, left, bottom, right
    
    # Create a copy of the input grid
    output_grid = input_grid.deep_copy()
    
    # Find the frame
    top, left, bottom, right = find_frame(input_grid)
    
    # Fill the grid sections
    for r in range(len(output_grid.values)):
        for c in range(len(output_grid.values[0])):
            if output_grid.values[r][c] != 0:
                continue
            
            if r < top or r > bottom or c < left or c > right:
                if r <= top and c <= left:
                    output_grid.values[r][c] = 2  # Top-left: red
                elif r <= top and c >= right:
                    output_grid.values[r][c] = 4  # Top-right: yellow
                elif r >= bottom and c <= left:
                    output_grid.values[r][c] = 1  # Bottom-left: blue
                elif r >= bottom and c >= right:
                    output_grid.values[r][c] = 8  # Bottom-right: sky blue
            elif r > top and r < bottom and c > left and c < right:
                output_grid.values[r][c] = 7  # Inside frame: orange
    
    return output_grid
