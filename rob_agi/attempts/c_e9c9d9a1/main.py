from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e9c9d9a1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling rectangles formed by green (3) lines.
    
    The solution:
    1. Identifies rectangles formed by green (3) horizontal lines.
    2. Fills these rectangles based on their position:
       - Top row: red (2) for leftmost, yellow (4) for rightmost, orange (7) for middle
       - Middle rows: all orange (7)
       - Bottom row: blue (1) for leftmost, sky blue (8) for rightmost, orange (7) for middle
    3. Preserves the green (3) lines and any non-black (0) cells from the input.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid according to the pattern.
    """
    def find_horizontal_lines(grid: ColoredGrid) -> List[int]:
        return [i for i, row in enumerate(grid.values) if all(cell == 3 for cell in row)]
    
    def identify_rectangles(grid: ColoredGrid, h_lines: List[int]) -> List[Tuple[int, int, int, int]]:
        rectangles = []
        for i in range(len(h_lines) - 1):
            top, bottom = h_lines[i], h_lines[i+1]
            left = 0
            for j in range(len(grid.values[0])):
                if grid.values[top][j] == 3:
                    if left != j:
                        rectangles.append((top, left, bottom, j))
                    left = j + 1
        return rectangles
    
    def fill_rectangle(grid: ColoredGrid, rect: Tuple[int, int, int, int], color: int) -> None:
        top, left, bottom, right = rect
        for r in range(top + 1, bottom):
            for c in range(left, right):
                if grid.values[r][c] == 0:
                    grid.values[r][c] = color
    
    # Create a copy of the input grid
    output_grid = input_grid.deep_copy()
    
    # Find horizontal green lines and identify rectangles
    h_lines = find_horizontal_lines(input_grid)
    rectangles = identify_rectangles(input_grid, h_lines)
    
    # Fill rectangles based on their position
    for i, rect in enumerate(rectangles):
        row = i // (len(rectangles) // len(h_lines))
        col = i % (len(rectangles) // len(h_lines))
        
        if row == 0:  # Top row
            color = 2 if col == 0 else 4 if col == len(rectangles) // len(h_lines) - 1 else 7
        elif row == len(h_lines) - 2:  # Bottom row
            color = 1 if col == 0 else 8 if col == len(rectangles) // len(h_lines) - 1 else 7
        else:  # Middle rows
            color = 7
        
        fill_rectangle(output_grid, rect, color)
    
    return output_grid
