from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import heapq

def solve_d47aa2ff(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 10x21 input grid into a 10x10 output grid by:
    1. Extracting the left 10x10 portion of the input grid.
    2. Preserving existing colors in the right half (columns 5-9).
    3. Adding two blue (1) dots and two red (2) dots to the right half.
    4. Placing the blue and red dots in balanced positions based on the existing pattern.

    The placement of new dots considers:
    - Vertical and horizontal centrality
    - Avoiding adjacency to existing colored squares
    - Placing red dots adjacent to blue dots
    - Balancing dot placement between upper and lower sections of the right half

    Args:
    input_grid (ColoredGrid): A 10x21 grid with a central gray (5) line.

    Returns:
    ColoredGrid: A 10x10 grid with the transformation applied.
    """
    # Step 1: Extract the left 10x10 portion of the input grid
    output_grid = ColoredGrid(values=[row[:10] for row in input_grid.values[:10]])
    
    # Step 2: Preserve existing colors in the right half
    for i in range(10):
        for j in range(5, 10):
            if input_grid.values[i][j] != 0 and input_grid.values[i][j] != 5:
                output_grid.values[i][j] = input_grid.values[i][j]
    
    # Step 3 & 4: Add blue and red dots to balanced positions
    empty_positions = [(i, j) for i in range(10) for j in range(5, 10) if output_grid.values[i][j] == 0]
    place_new_dots(output_grid, empty_positions)
    
    return output_grid

def place_new_dots(grid: ColoredGrid, empty_positions: List[Tuple[int, int]]):
    blue_positions = []
    red_positions = []
    
    # Place blue dots
    for _ in range(2):
        if empty_positions:
            best_pos = max(empty_positions, key=lambda pos: calculate_balance_score(grid, pos[0], pos[1]))
            grid.values[best_pos[0]][best_pos[1]] = 1  # Place blue dot
            blue_positions.append(best_pos)
            empty_positions.remove(best_pos)
    
    # Place red dots
    for blue_pos in blue_positions:
        adjacent_positions = get_adjacent_positions(blue_pos[0], blue_pos[1])
        valid_red_positions = [pos for pos in adjacent_positions if pos in empty_positions]
        if valid_red_positions:
            best_pos = max(valid_red_positions, key=lambda pos: calculate_balance_score(grid, pos[0], pos[1]))
            grid.values[best_pos[0]][best_pos[1]] = 2  # Place red dot
            red_positions.append(best_pos)
            empty_positions.remove(best_pos)

def get_adjacent_positions(row: int, col: int) -> List[Tuple[int, int]]:
    return [(row+dr, col+dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= row+dr < 10 and 5 <= col+dc < 10]

def calculate_balance_score(grid: ColoredGrid, row: int, col: int) -> float:
    score = 0
    # Vertical centrality
    vertical_score = 5 - abs(row - 4.5)
    score += vertical_score * 2
    
    # Horizontal centrality
    horizontal_score = 2 - abs(col - 7)
    score += horizontal_score
    
    # Penalty for adjacent colored squares
    for adj_row, adj_col in get_adjacent_positions(row, col):
        if grid.values[adj_row][adj_col] != 0:
            score -= 2
    
    # Bonus for being in the less populated half
    upper_count = sum(1 for i in range(5) for j in range(5, 10) if grid.values[i][j] != 0)
    lower_count = sum(1 for i in range(5, 10) for j in range(5, 10) if grid.values[i][j] != 0)
    if (row < 5 and upper_count <= lower_count) or (row >= 5 and lower_count < upper_count):
        score += 3
    
    return score
