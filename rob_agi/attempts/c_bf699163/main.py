from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_bf699163(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the bf699163 challenge by finding all valid 3x3 patterns in the input grid
    and returning a new 3x3 grid based on the pattern with the lowest color value.

    A valid pattern is a 3x3 subgrid with a gray (5) center and all surrounding cells
    of the same non-gray color. The function returns a new 3x3 grid with the pattern
    that has the lowest surrounding color value.

    Args:
    input_grid (ColoredGrid): The input grid to analyze.

    Returns:
    ColoredGrid: A 3x3 grid representing the valid pattern with the lowest color value,
                 or None if no valid pattern is found.
    """
    def find_valid_patterns(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
        valid_patterns = []
        rows, cols = grid.get_dimensions()
        
        for row in range(rows):
            for col in range(cols):
                if grid.values[row][col] == 5:  # Center must be gray
                    surrounding_color = None
                    is_valid = True
                    
                    for i in range(max(0, row-1), min(rows, row+2)):
                        for j in range(max(0, col-1), min(cols, col+2)):
                            if i == row and j == col:
                                continue
                            current_color = grid.values[i][j]
                            if current_color == 5:  # Surrounding cells can't be gray
                                is_valid = False
                                break
                            if surrounding_color is None:
                                surrounding_color = current_color
                            elif current_color != surrounding_color:
                                is_valid = False
                                break
                        if not is_valid:
                            break
                    
                    if is_valid and surrounding_color is not None:
                        valid_patterns.append((surrounding_color, row, col))
        
        return valid_patterns

    valid_patterns = find_valid_patterns(input_grid)
    
    if not valid_patterns:
        return None  # No valid pattern found
    
    # Sort patterns by color value and select the one with the lowest color
    selected_pattern = min(valid_patterns)
    color = selected_pattern[0]
    
    # Create and return the new 3x3 ColoredGrid
    return ColoredGrid(values=[
        [color, color, color],
        [color, 5, color],
        [color, color, color]
    ])
