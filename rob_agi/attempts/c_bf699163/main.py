from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import Counter

def solve_bf699163(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the bf699163 challenge by finding the most isolated valid 3x3 pattern in the input grid.

    A valid pattern is a 3x3 subgrid with a gray (5) center and all surrounding cells
    of the same non-gray color. The most isolated pattern is determined by the rarity
    of its color in the entire grid. If multiple patterns are equally isolated,
    the one with the lowest color value is chosen.

    Args:
    input_grid (ColoredGrid): The input grid to analyze.

    Returns:
    ColoredGrid: A 3x3 grid representing the most isolated valid pattern,
                 or None if no valid pattern is found.
    """
    def find_valid_patterns(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
        valid_patterns = []
        rows, cols = grid.get_dimensions()
        
        for row in range(1, rows - 1):
            for col in range(1, cols - 1):
                if grid.values[row][col] == 5:  # Center must be gray
                    surrounding_color = grid.values[row-1][col]
                    if surrounding_color != 5 and all(
                        grid.values[r][c] == surrounding_color
                        for r in range(row-1, row+2)
                        for c in range(col-1, col+2)
                        if (r, c) != (row, col)
                    ):
                        valid_patterns.append((surrounding_color, row, col))
        
        return valid_patterns

    def count_colors(grid: ColoredGrid) -> Dict[int, int]:
        return Counter(cell for row in grid.values for cell in row if cell != 5)

    valid_patterns = find_valid_patterns(input_grid)
    
    if not valid_patterns:
        return None  # No valid pattern found
    
    color_counts = count_colors(input_grid)
    min_count = min(color_counts[color] for color, _, _ in valid_patterns)
    
    most_isolated_patterns = [
        (color, row, col) for color, row, col in valid_patterns
        if color_counts[color] == min_count
    ]
    
    selected_pattern = min(most_isolated_patterns, key=lambda x: x[0])
    color = selected_pattern[0]
    
    # Create and return the new 3x3 ColoredGrid
    return ColoredGrid(values=[
        [color, color, color],
        [color, 5, color],
        [color, color, color]
    ])
