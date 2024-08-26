from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_34b99a2b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the 5x9 input grid into a 5x4 output grid based on the following steps:
    1. Splits the input into left and right halves, analyzing sky (8) and gray (5) regions.
    2. Divides each half into quadrants and calculates color density.
    3. Marks quadrants with high density and fills corresponding areas in the output grid.
    4. Applies flow rules to connect marked quadrants.
    5. Emphasizes the bottom row based on high-density bottom quadrants.
    6. Ensures vertical consistency in columns with multiple red cells.
    7. Removes isolated red cells and makes final adjustments to the bottom row.
    """
    left_half, right_half = split_grid(input_grid)
    output = [[0 for _ in range(4)] for _ in range(5)]
    
    process_half(left_half, output, 0, 2, 8)
    process_half(right_half, output, 2, 4, 5)
    
    apply_flow_rules(output)
    apply_bottom_emphasis(left_half, right_half, output)
    ensure_vertical_consistency(output)
    remove_isolated_cells(output)
    adjust_bottom_row(output)
    
    return ColoredGrid(values=output)

def split_grid(grid: ColoredGrid) -> Tuple[List[List[int]], List[List[int]]]:
    left_half = [row[:4] for row in grid.values]
    right_half = [row[5:] for row in grid.values]
    return left_half, right_half

def process_half(half: List[List[int]], output: List[List[int]], start_col: int, end_col: int, color: int):
    quadrants = [
        (0, 0), (0, 2), (2, 0), (2, 2)
    ]
    for top, left in quadrants:
        if calculate_density(half, top, left, color) > 0.3:
            fill_output_quadrant(output, top, start_col + (left // 2), 2)

def calculate_density(grid: List[List[int]], top: int, left: int, color: int) -> float:
    count = sum(1 for r in range(top, top+3) for c in range(left, left+2) if grid[r][c] == color)
    return count / 6

def fill_output_quadrant(output: List[List[int]], top: int, left: int, size: int):
    for r in range(top, top+size):
        for c in range(left, left+size):
            output[r][c] = 2

def apply_flow_rules(output: List[List[int]]):
    # Horizontal flow
    for r in range(5):
        if output[r][0] == 2 and output[r][2] == 2:
            output[r][1] = 2
    # Vertical flow
    for c in range(4):
        for r in range(4):
            if output[r][c] == 2 and output[r+1][c] == 2:
                output[r][c] = output[r+1][c] = 2
    # Diagonal flow
    for r in range(4):
        for c in range(3):
            if output[r][c] == 2 and output[r+1][c+1] == 2:
                output[r][c+1] = output[r+1][c] = 2

def apply_bottom_emphasis(left_half: List[List[int]], right_half: List[List[int]], output: List[List[int]]):
    if calculate_density(left_half, 2, 0, 8) > 0.5:
        output[4][0] = output[4][1] = 2
    if calculate_density(right_half, 2, 0, 5) > 0.5:
        output[4][2] = output[4][3] = 2

def ensure_vertical_consistency(output: List[List[int]]):
    for c in range(4):
        if sum(output[r][c] for r in range(5)) >= 2:
            for r in range(5):
                output[r][c] = 2

def remove_isolated_cells(output: List[List[int]]):
    for r in range(5):
        for c in range(4):
            if output[r][c] == 2:
                neighbors = sum(1 for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]
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
