from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_d47aa2ff(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 10x21 input grid into a 10x10 output grid by:
    1. Extracting the left 10x10 portion of the input grid.
    2. Preserving existing dots in the right half (columns 5-9).
    3. Adding two blue (1) dots and two red (2) dots to the right half.
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
    upper_section = [(i, j) for i in range(3, 6) for j in range(5, 10) if output_grid.values[i][j] == 0]
    lower_section = [(i, j) for i in range(6, 10) for j in range(5, 10) if output_grid.values[i][j] == 0]
    
    place_dot_pair(output_grid, upper_section)
    place_dot_pair(output_grid, lower_section)
    
    return output_grid

def place_dot_pair(grid: ColoredGrid, section: List[Tuple[int, int]]):
    if not section:
        return
    
    blue_pos = max(section, key=lambda pos: calculate_balance_score(grid, pos[0], pos[1]))
    grid.values[blue_pos[0]][blue_pos[1]] = 1  # Place blue dot
    
    adjacent_positions = get_adjacent_positions(blue_pos[0], blue_pos[1])
    valid_red_positions = [pos for pos in adjacent_positions if pos in section and grid.values[pos[0]][pos[1]] == 0]
    
    if valid_red_positions:
        red_pos = max(valid_red_positions, key=lambda pos: calculate_balance_score(grid, pos[0], pos[1]))
        grid.values[red_pos[0]][red_pos[1]] = 2  # Place red dot

def get_adjacent_positions(row: int, col: int) -> List[Tuple[int, int]]:
    return [(row+dr, col+dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] 
            if (dr != 0 or dc != 0) and 0 <= row+dr < 10 and 5 <= col+dc < 10]

def calculate_balance_score(grid: ColoredGrid, row: int, col: int) -> float:
    score = 0
    # Vertical centrality
    vertical_score = 5 - abs(row - 6.5)
    score += vertical_score * 2
    
    # Horizontal centrality
    horizontal_score = 4 - abs(col - 7)
    score += horizontal_score
    
    # Penalty for adjacent colored squares
    for adj_row, adj_col in get_adjacent_positions(row, col):
        if grid.values[adj_row][adj_col] != 0:
            score -= 1
    
    return score
