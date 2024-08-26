from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_34b99a2b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 5x9 input grid into a 5x4 output grid based on the following steps:
    1. Splits the input grid into left and right halves.
    2. Identifies vertical lines and diagonal patterns of sky (8) and gray (5) colors.
    3. Creates a 5x4 output grid with red (2) squares based on the identified patterns.
    4. Processes the bottom row to reflect significant activity in the input.
    5. Ensures connectivity and applies smoothing rules.
    6. Makes final adjustments to balance the pattern.
    """
    left_half, right_half = split_grid(input_grid)
    output = [[0 for _ in range(4)] for _ in range(5)]
    
    process_vertical_lines(left_half, right_half, output)
    process_diagonal_patterns(left_half, right_half, output)
    process_bottom_row(left_half, right_half, output)
    
    ensure_connectivity(output)
    apply_smoothing_rules(output)
    final_adjustments(output)
    
    return ColoredGrid(values=output)

def split_grid(grid: ColoredGrid) -> Tuple[List[List[int]], List[List[int]]]:
    return [row[:4] for row in grid.values], [row[5:] for row in grid.values]

def process_vertical_lines(left_half: List[List[int]], right_half: List[List[int]], output: List[List[int]]):
    for c in range(4):
        if sum(left_half[r][c] == 8 for r in range(5)) >= 2 or sum(right_half[r][c] == 5 for r in range(5)) >= 2:
            for r in range(5):
                output[r][c] = 2

def process_diagonal_patterns(left_half: List[List[int]], right_half: List[List[int]], output: List[List[int]]):
    for r in range(4):
        for c in range(3):
            if (left_half[r][c] == 8 and left_half[r+1][c+1] == 8) or (right_half[r][c] == 5 and right_half[r+1][c+1] == 5):
                output[r][c] = output[r+1][c+1] = 2

def process_bottom_row(left_half: List[List[int]], right_half: List[List[int]], output: List[List[int]]):
    bottom_activity = sum(left_half[4][c] == 8 or right_half[4][c] == 5 for c in range(4))
    for c in range(4):
        if bottom_activity >= 2 and output[4][c] == 0:
            output[4][c] = 2

def ensure_connectivity(output: List[List[int]]):
    def get_neighbors(r, c):
        return [(r+dr, c+dc) for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]
                if 0 <= r+dr < 5 and 0 <= c+dc < 4]
    
    def dfs(r, c, visited):
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited:
                visited.add((curr_r, curr_c))
                for nr, nc in get_neighbors(curr_r, curr_c):
                    if output[nr][nc] == 2:
                        stack.append((nr, nc))
    
    red_cells = [(r, c) for r in range(5) for c in range(4) if output[r][c] == 2]
    if not red_cells:
        return
    
    visited = set()
    dfs(red_cells[0][0], red_cells[0][1], visited)
    
    for r, c in red_cells:
        if (r, c) not in visited:
            nearest = min(visited, key=lambda x: abs(x[0]-r) + abs(x[1]-c))
            nr, nc = nearest
            for cr, cc in [(r, nc), (nr, c)]:
                output[cr][cc] = 2
            dfs(r, c, visited)

def apply_smoothing_rules(output: List[List[int]]):
    for _ in range(2):
        for r in range(5):
            for c in range(4):
                if output[r][c] == 2:
                    for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        if 0 <= r+dr < 5 and 0 <= c+dc < 4 and output[r+dr][c+dc] == 2:
                            output[r][c+dc] = output[r+dr][c] = 2
                    if r > 0 and r < 4 and output[r-1][c] == 2 and output[r+1][c] == 2:
                        output[r][c] = 2

def final_adjustments(output: List[List[int]]):
    total_red = sum(sum(row) for row in output) // 2
    if total_red < 8:
        for r in range(5):
            for c in range(4):
                if output[r][c] == 0 and any(output[nr][nc] == 2 for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)] if 0 <= nr < 5 and 0 <= nc < 4):
                    output[r][c] = 2
                    total_red += 1
                    if total_red == 8:
                        return
    elif total_red > 12:
        for r in range(5):
            for c in range(4):
                if output[r][c] == 2 and sum(output[nr][nc] == 2 for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)] if 0 <= nr < 5 and 0 <= nc < 4) <= 1:
                    output[r][c] = 0
                    total_red -= 1
                    if total_red == 12:
                        return
