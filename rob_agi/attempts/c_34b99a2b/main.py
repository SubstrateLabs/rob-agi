from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_34b99a2b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the 5x9 input grid into a 5x4 output grid based on the following steps:
    1. Splits the input into left and right halves, analyzing sky (8) and gray (5) regions.
    2. Identifies vertical and diagonal lines in each half.
    3. Translates identified patterns to the output grid.
    4. Applies smoothing rules to connect nearby red cells.
    5. Ensures vertical consistency in columns with multiple red cells.
    6. Processes the bottom row based on input density.
    7. Makes final adjustments to remove isolated cells and balance the bottom row.
    """
    left_half, right_half = split_grid(input_grid)
    output = [[0 for _ in range(4)] for _ in range(5)]
    
    process_half(left_half, output, 0, 2, 8)
    process_half(right_half, output, 2, 4, 5)
    
    apply_smoothing_rules(output)
    ensure_vertical_consistency(output)
    process_bottom_row(left_half, right_half, output)
    remove_isolated_cells(output)
    adjust_bottom_row(output)
    
    return ColoredGrid(values=output)

def split_grid(grid: ColoredGrid) -> Tuple[List[List[int]], List[List[int]]]:
    left_half = [row[:4] for row in grid.values]
    right_half = [row[5:] for row in grid.values]
    return left_half, right_half

def process_half(half: List[List[int]], output: List[List[int]], start_col: int, end_col: int, color: int):
    vertical_lines = identify_vertical_lines(half, color)
    diagonal_lines = identify_diagonal_lines(half, color)
    
    for col in vertical_lines:
        output_col = start_col + (col // 2)
        for r in range(5):
            output[r][output_col] = 2
    
    for start_r, start_c in diagonal_lines:
        output_r = start_r
        output_c = start_col + (start_c // 2)
        while output_r < 5 and output_c < end_col:
            output[output_r][output_c] = 2
            output_r += 1
            output_c += 1

def identify_vertical_lines(half: List[List[int]], color: int) -> List[int]:
    return [c for c in range(4) if sum(1 for r in range(5) if half[r][c] == color) >= 2]

def identify_diagonal_lines(half: List[List[int]], color: int) -> List[Tuple[int, int]]:
    diagonals = []
    for r in range(4):
        for c in range(3):
            if half[r][c] == color and half[r+1][c+1] == color:
                diagonals.append((r, c))
    return diagonals

def apply_smoothing_rules(output: List[List[int]]):
    # Vertical smoothing
    for c in range(4):
        for r in range(1, 4):
            if output[r-1][c] == 2 and output[r+1][c] == 2:
                output[r][c] = 2
    
    # Diagonal smoothing
    for r in range(4):
        for c in range(3):
            if output[r][c] == 2 and output[r+1][c+1] == 2:
                output[r][c+1] = output[r+1][c] = 2

def ensure_vertical_consistency(output: List[List[int]]):
    for c in range(4):
        if sum(output[r][c] for r in range(5)) >= 2:
            for r in range(5):
                output[r][c] = 2

def process_bottom_row(left_half: List[List[int]], right_half: List[List[int]], output: List[List[int]]):
    left_density = sum(1 for r in range(2, 5) for c in range(4) if left_half[r][c] == 8) / 12
    right_density = sum(1 for r in range(2, 5) for c in range(4) if right_half[r][c] == 5) / 12
    
    if left_density > 0.5:
        output[4][0] = output[4][1] = 2
    if right_density > 0.5:
        output[4][2] = output[4][3] = 2

def remove_isolated_cells(output: List[List[int]]):
    for r in range(5):
        for c in range(4):
            if output[r][c] == 2:
                neighbors = sum(1 for dr, dc in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]
                                if 0 <= r+dr < 5 and 0 <= c+dc < 4 and output[r+dr][c+dc] == 2)
                if neighbors == 0:
                    output[r][c] = 0

def adjust_bottom_row(output: List[List[int]]):
    bottom_red = sum(output[4])
    if 0 < bottom_red < 4:
        for c in range(4):
            if output[4][c] == 0 and output[3][c] == 2:
                output[4][c] = 2
        if sum(output[4]) == 4:
            return
        for c in range(4):
            if output[4][c] == 2 and output[3][c] == 0:
                output[4][c] = 0
