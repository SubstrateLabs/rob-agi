from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_5a5a2103(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a 4x4 pattern to each section of the grid.
    
    The transformation works as follows:
    1. Identifies the dividing lines in the grid.
    2. For each row of sections:
       a. Finds the first non-zero, non-dividing-line color in the leftmost section.
       b. If a color is found, generates a 4x4 pattern for this color.
       c. Applies this pattern across the entire row, respecting dividing lines.
       d. If no color is found, leaves the row unchanged.
    3. Preserves the original dividing lines in the output.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    
    def find_dividing_lines(grid: List[List[int]]) -> Tuple[int, List[int], List[int]]:
        divider_color = next(color for row in grid for color in row if color != 0)
        horizontal_lines = [i for i, row in enumerate(grid) if all(cell == divider_color for cell in row)]
        vertical_lines = [j for j in range(len(grid[0])) if all(row[j] == divider_color for row in grid)]
        return divider_color, horizontal_lines, vertical_lines
    
    def generate_pattern(color: int) -> List[List[int]]:
        return [
            [color, color, 0, color],
            [0, color, color, 0],
            [color, color, color, color],
            [color, 0, 0, color]
        ]
    
    def apply_pattern(grid: List[List[int]], pattern: List[List[int]], start_row: int,
                      divider_color: int, vertical_lines: List[int]) -> None:
        for i in range(4):
            for col in range(len(grid[0])):
                if col not in vertical_lines and grid[start_row + i][col] != divider_color:
                    grid[start_row + i][col] = pattern[i][col % 4]
    
    # Find dividing lines
    divider_color, horizontal_lines, vertical_lines = find_dividing_lines(input_grid.values)
    
    # Create a new grid with the same dimensions as the input
    new_grid = [row[:] for row in input_grid.values]
    
    # Process each row of sections
    for section_start in range(0, len(new_grid), 5):
        if section_start in horizontal_lines:
            continue
    
        # Find the color for this row of sections
        section_color = next((color for color in new_grid[section_start] if color not in [0, divider_color]), None)
        
        if section_color is not None:
            pattern = generate_pattern(section_color)
            
            # Apply the pattern across the row
            for row in range(section_start, min(section_start + 4, len(new_grid))):
                apply_pattern(new_grid, pattern, row, divider_color, vertical_lines)
    
    return ColoredGrid(values=new_grid)
