from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e9c9d9a1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling sections formed by green (3) lines.
    
    The solution:
    1. Identifies key horizontal and vertical green (3) lines.
    2. Defines nine sections of the grid.
    3. Fills these sections based on their position:
       - Top-left: red (2)
       - Top-right: yellow (4)
       - Bottom-left: blue (1)
       - Bottom-right: sky blue (8)
       - Middle-center: orange (7)
       - Edge sections (top-center, middle-left, middle-right, bottom-center): remain black (0)
    4. Preserves all green (3) lines and any non-black (0) cells from the input.
    5. Handles edge cases where there might not be enough green lines to fully define all sections.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid according to the pattern.
    """
    def find_key_lines(grid: ColoredGrid) -> Tuple[List[int], List[int]]:
        rows, cols = len(grid.values), len(grid.values[0])
        h_lines = [i for i, row in enumerate(grid.values) if any(cell == 3 for cell in row)]
        v_lines = [j for j in range(cols) if any(row[j] == 3 for row in grid.values)]
        
        if len(h_lines) < 2:
            h_lines = [0] + h_lines + [rows - 1]
        if len(v_lines) < 2:
            v_lines = [0] + v_lines + [cols - 1]
        
        return (
            [h_lines[0], h_lines[1], h_lines[-2], h_lines[-1]],
            [v_lines[0], v_lines[1], v_lines[-2], v_lines[-1]]
        )
    
    def fill_section(grid: ColoredGrid, top: int, left: int, bottom: int, right: int, color: int) -> None:
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if grid.values[r][c] == 0:
                    grid.values[r][c] = color
    
    # Create a copy of the input grid
    output_grid = input_grid.deep_copy()
    
    # Find key horizontal and vertical green lines
    h_lines, v_lines = find_key_lines(input_grid)
    
    # Define the nine sections and fill them
    sections = [
        ((h_lines[0], v_lines[0]), (h_lines[1], v_lines[1]), 2),  # Top-left: red
        ((h_lines[0], v_lines[2]), (h_lines[1], v_lines[3]), 4),  # Top-right: yellow
        ((h_lines[2], v_lines[0]), (h_lines[3], v_lines[1]), 1),  # Bottom-left: blue
        ((h_lines[2], v_lines[2]), (h_lines[3], v_lines[3]), 8),  # Bottom-right: sky blue
        ((h_lines[1], v_lines[1]), (h_lines[2], v_lines[2]), 7),  # Middle-center: orange
    ]
    
    for (top, left), (bottom, right), color in sections:
        fill_section(output_grid, top, left, bottom, right, color)
    
    # Preserve original non-black, non-green cells
    for r in range(len(input_grid.values)):
        for c in range(len(input_grid.values[0])):
            if input_grid.values[r][c] not in [0, 3]:
                output_grid.values[r][c] = input_grid.values[r][c]
    
    return output_grid
