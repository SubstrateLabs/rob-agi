from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_d47aa2ff(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 10x21 input grid into a 10x10 output grid by:
    1. Extracting the left 10x10 portion of the input grid.
    2. Preserving existing dots in the right half (columns 5-9).
    3. Adding one blue (1) dot and one red (2) dot to the right half.
    4. Placing the blue and red dots in balanced positions based on the existing pattern.

    Args:
    input_grid (ColoredGrid): A 10x21 grid with a central gray (5) line.

    Returns:
    ColoredGrid: A 10x10 grid with the transformation applied.
    """
    # Step 1: Extract the left 10x10 portion of the input grid
    output_grid = ColoredGrid(values=[row[:10] for row in input_grid.values[:10]])
    
    # Step 2: Preserve existing dots in the right half
    for i in range(10):
        for j in range(5, 10):
            if input_grid.values[i][j] != 0 and input_grid.values[i][j] != 5:
                output_grid.values[i][j] = input_grid.values[i][j]
    
    # Step 3 & 4: Add blue and red dots to balanced positions
    blue_pos = find_optimal_position(output_grid)
    output_grid.values[blue_pos[0]][blue_pos[1]] = 1  # Place blue dot
    
    red_pos = find_red_position(output_grid, blue_pos)
    output_grid.values[red_pos[0]][red_pos[1]] = 2  # Place red dot
    
    return output_grid

def find_optimal_position(grid: ColoredGrid) -> Tuple[int, int]:
    best_score = float('-inf')
    best_pos = None
    for i in range(3, 7):
        for j in range(5, 10):
            if grid.values[i][j] == 0:
                score = calculate_balance_score(grid, i, j)
                if score > best_score:
                    best_score = score
                    best_pos = (i, j)
    return best_pos

def calculate_balance_score(grid: ColoredGrid, row: int, col: int) -> float:
    score = 0
    # Distance from other dots
    for i in range(10):
        for j in range(5, 10):
            if grid.values[i][j] != 0:
                distance = abs(i - row) + abs(j - col)
                score += 1 / (distance + 1)  # Avoid division by zero
    
    # Vertical centrality
    vertical_score = 4 - abs(row - 4.5)
    score += vertical_score * 2
    
    # Horizontal position
    horizontal_score = 3 - abs(col - 7)
    score += horizontal_score
    
    return score

def find_red_position(grid: ColoredGrid, blue_pos: Tuple[int, int]) -> Tuple[int, int]:
    row, col = blue_pos
    diagonals = [(row-1, col+1), (row+1, col+1), (row-1, col-1), (row+1, col-1)]
    adjacents = [(row-1, col), (row+1, col), (row, col+1), (row, col-1)]
    
    for r, c in diagonals + adjacents:
        if 0 <= r < 10 and 5 <= c < 10 and grid.values[r][c] == 0:
            return (r, c)
    
    # If no suitable position found, return a default position
    return (row, col+1) if col < 9 else (row, col-1)
