from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_bf699163(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the bf699163 challenge by finding all valid 3x3 patterns in the input grid
    and returning a new 3x3 grid based on the most isolated pattern with the lowest color value.

    A valid pattern is a 3x3 subgrid with a gray (5) center and all surrounding cells
    of the same non-gray color. The function calculates an isolation score for each valid pattern
    by counting the number of gray cells in the 16 cells surrounding the 3x3 pattern.
    It then selects the pattern with the highest isolation score, or if there are multiple
    patterns with the same highest score, it chooses the one with the lowest color value.

    Args:
    input_grid (ColoredGrid): The input grid to analyze.

    Returns:
    ColoredGrid: A 3x3 grid representing the most isolated valid pattern with the lowest color value,
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
                    
                    for i in range(row-1, row+2):
                        for j in range(col-1, col+2):
                            if i == row and j == col:
                                continue
                            if 0 <= i < rows and 0 <= j < cols:
                                current_color = grid.values[i][j]
                                if current_color == 5:  # Surrounding cells can't be gray
                                    is_valid = False
                                    break
                                if surrounding_color is None:
                                    surrounding_color = current_color
                                elif current_color != surrounding_color:
                                    is_valid = False
                                    break
                            else:
                                is_valid = False
                                break
                        if not is_valid:
                            break
                    
                    if is_valid and surrounding_color is not None:
                        valid_patterns.append((surrounding_color, row, col))
        
        return valid_patterns

    def calculate_isolation_score(grid: ColoredGrid, row: int, col: int) -> int:
        rows, cols = grid.get_dimensions()
        score = 0
        for i in range(row-2, row+3):
            for j in range(col-2, col+3):
                if (row-1 <= i <= row+1 and col-1 <= j <= col+1) or i < 0 or i >= rows or j < 0 or j >= cols:
                    continue
                if grid.values[i][j] == 5:
                    score += 1
        return score

    valid_patterns = find_valid_patterns(input_grid)
    
    if not valid_patterns:
        return None  # No valid pattern found
    
    # Calculate isolation scores and find the most isolated pattern(s)
    isolation_scores = [(pattern, calculate_isolation_score(input_grid, pattern[1], pattern[2])) for pattern in valid_patterns]
    max_isolation_score = max(score for _, score in isolation_scores)
    most_isolated_patterns = [pattern for pattern, score in isolation_scores if score == max_isolation_score]
    
    # Select the pattern with the lowest color value among the most isolated patterns
    selected_pattern = min(most_isolated_patterns, key=lambda x: x[0])
    color = selected_pattern[0]
    
    # Create and return the new 3x3 ColoredGrid
    return ColoredGrid(values=[
        [color, color, color],
        [color, 5, color],
        [color, color, color]
    ])
