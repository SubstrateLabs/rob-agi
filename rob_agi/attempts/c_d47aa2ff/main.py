from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_d47aa2ff(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 10x21 input grid into a 10x10 output grid by:
    1. Extracting the left 10x10 portion of the input grid.
    2. Preserving existing dots in the right half (columns 5-9).
    3. Adding one blue (1) dot and one red (2) dot to the right half.
    4. Placing the blue and red dots in optimal positions based on the existing pattern.

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
    
    # Analyze the pattern and find optimal placement areas
    emptiness_scores = calculate_emptiness_scores(output_grid)
    
    # Step 3 & 4: Add blue and red dots to optimal positions
    blue_pos, red_pos = find_optimal_positions(emptiness_scores)
    
    output_grid.values[blue_pos[0]][blue_pos[1]] = 1  # Place blue dot
    output_grid.values[red_pos[0]][red_pos[1]] = 2  # Place red dot
    
    return output_grid

def calculate_emptiness_scores(grid: ColoredGrid) -> List[List[int]]:
    scores = [[0 for _ in range(5)] for _ in range(10)]
    for i in range(10):
        for j in range(5, 10):
            if grid.values[i][j] == 0:
                scores[i][j-5] = calculate_score(grid, i, j)
    return scores

def calculate_score(grid: ColoredGrid, row: int, col: int) -> int:
    score = 0
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if 0 <= row + di < 10 and 0 <= col + dj < 10:
                if grid.values[row+di][col+dj] == 0:
                    score += 1
    return score

def find_optimal_positions(scores: List[List[int]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    flat_scores = [(i, j, scores[i][j]) for i in range(10) for j in range(5)]
    sorted_positions = sorted(flat_scores, key=lambda x: x[2], reverse=True)
    
    blue_pos = (sorted_positions[0][0], sorted_positions[0][1] + 5)
    red_pos = (sorted_positions[1][0], sorted_positions[1][1] + 5)
    
    return blue_pos, red_pos
