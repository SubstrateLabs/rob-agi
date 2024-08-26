from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_34b99a2b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the 5x9 input grid into a 5x4 output grid based on the following steps:
    1. Analyzes the input grid for patterns of sky (8) and gray (5) squares.
    2. Creates a 5x4 output grid with red (2) squares based on the input patterns.
    3. Ensures a coherent pattern of red squares, focusing on the top half of the grid.
    4. Applies rules for vertical consistency and bottom row processing.
    5. Makes final adjustments to balance the pattern and match example outputs.
    """
    output = [[0 for _ in range(4)] for _ in range(5)]
    
    analyze_input_and_create_output(input_grid, output)
    ensure_coherent_pattern(output)
    apply_vertical_consistency(output)
    process_bottom_row(input_grid, output)
    final_adjustments(output)
    
    return ColoredGrid(values=output)

def analyze_input_and_create_output(input_grid: ColoredGrid, output: List[List[int]]):
    for r in range(5):
        sky_count = sum(1 for c in range(3) if input_grid.values[r][c] == 8)
        gray_count = sum(1 for c in range(5, 9) if input_grid.values[r][c] == 5)
        if sky_count >= 2 or gray_count >= 2:
            output[r][0] = output[r][1] = 2
        if input_grid.values[r][4] == 4:  # Yellow column
            output[r][2] = 2

def split_grid(grid: ColoredGrid) -> Tuple[List[List[int]], List[List[int]]]:
    left_half = [row[:4] for row in grid.values]
    right_half = [row[5:] for row in grid.values]
    return left_half, right_half

def calculate_density_scores(left_half: List[List[int]], right_half: List[List[int]]) -> List[float]:
    scores = []
    for half, color in [(left_half, 8), (right_half, 5)]:
        for rows, cols in [(slice(0, 3), slice(0, 2)), (slice(0, 3), slice(2, 4)), 
                           (slice(3, 5), slice(0, 2)), (slice(3, 5), slice(2, 4))]:
            quadrant = [row[cols] for row in half[rows]]
            score = sum(cell == color for row in quadrant for cell in row) / len(quadrant) / len(quadrant[0])
            scores.append(score)
    return scores

def map_inverse_density_to_output(scores: List[float], output: List[List[int]], total_red: int):
    inverse_scores = [1 - score for score in scores]
    total_inverse = sum(inverse_scores)
    quadrants = [(3, 2), (3, 0), (0, 2), (0, 0)]  # Inverse mapping
    remaining_red = total_red
    
    for score, (start_row, start_col) in zip(inverse_scores, quadrants):
        num_red = int((score / total_inverse) * total_red)
        num_red = min(num_red, remaining_red, 4)  # Ensure we don't place too many reds
        cells = [(r, c) for r in range(start_row, start_row + 2) for c in range(start_col, start_col + 2)]
        for _ in range(num_red):
            if cells:
                r, c = cells.pop(random.randint(0, len(cells) - 1))
                output[r][c] = 2
                remaining_red -= 1
    
    # Place any remaining reds
    while remaining_red > 0:
        r, c = random.randint(0, 4), random.randint(0, 3)
        if output[r][c] == 0:
            output[r][c] = 2
            remaining_red -= 1

def identify_vertical_lines(quadrant: List[List[int]], color: int) -> List[int]:
    return [c for c in range(len(quadrant[0])) if sum(1 for r in range(len(quadrant)) if quadrant[r][c] == color) >= 2]

def identify_diagonal_lines(quadrant: List[List[int]], color: int) -> List[Tuple[int, int]]:
    diagonals = []
    for r in range(len(quadrant) - 1):
        for c in range(len(quadrant[0]) - 1):
            if quadrant[r][c] == color and quadrant[r+1][c+1] == color:
                diagonals.append((r, c))
    return diagonals

def apply_smoothing_rules(output: List[List[int]]):
    for _ in range(2):  # Apply smoothing twice
        for r in range(5):
            for c in range(4):
                if output[r][c] == 2:
                    # Connect diagonal red squares
                    for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        if 0 <= r+dr < 5 and 0 <= c+dc < 4 and output[r+dr][c+dc] == 2:
                            output[r][c+dc] = output[r+dr][c] = 2
                    # Fill gaps between vertical red squares
                    if r > 0 and r < 4 and output[r-1][c] == 2 and output[r+1][c] == 2:
                        output[r][c] = 2

def ensure_vertical_consistency(output: List[List[int]]):
    for c in range(4):
        red_count = sum(output[r][c] == 2 for r in range(5))
        if red_count >= 3:
            for r in range(5):
                output[r][c] = 2
        elif red_count == 2:
            gaps = [r for r in range(1, 4) if output[r-1][c] == 2 and output[r+1][c] == 2 and output[r][c] == 0]
            for r in gaps:
                output[r][c] = 2

def process_bottom_row(left_half: List[List[int]], right_half: List[List[int]], output: List[List[int]]):
    bottom_activity = (
        sum(1 for r in range(3, 5) for c in range(4) if left_half[r][c] == 8) +
        sum(1 for r in range(3, 5) for c in range(4) if right_half[r][c] == 5)
    ) / 16
    num_red_bottom = int(bottom_activity * 4)
    for c in range(4):
        if output[3][c] == 2 or (num_red_bottom > 0 and output[4][c] == 0):
            output[4][c] = 2
            num_red_bottom -= 1

def remove_isolated_cells(output: List[List[int]]):
    for r in range(5):
        for c in range(4):
            if output[r][c] == 2:
                neighbors = sum(1 for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]
                                if 0 <= r+dr < 5 and 0 <= c+dc < 4 and output[r+dr][c+dc] == 2)
                if neighbors == 0:
                    output[r][c] = 0

def final_adjustments(output: List[List[int]]):
    # Balance the pattern
    total_red = sum(sum(row) for row in output) // 2
    if total_red > 10:  # If there are too many red squares, remove some
        for r in range(5):
            for c in range(4):
                if output[r][c] == 2 and sum(output[r]) > 2:
                    output[r][c] = 0
                    total_red -= 1
                    if total_red <= 10:
                        return
    elif total_red < 6:  # If there are too few red squares, add some
        for r in range(5):
            for c in range(4):
                if output[r][c] == 0 and (r > 0 and output[r-1][c] == 2 or c > 0 and output[r][c-1] == 2):
                    output[r][c] = 2
                    total_red += 1
                    if total_red >= 6:
                        return
import random

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
            # Connect isolated red cell to the nearest connected red cell
            min_dist = float('inf')
            nearest = None
            for vr, vc in visited:
                dist = abs(r - vr) + abs(c - vc)
                if dist < min_dist:
                    min_dist = dist
                    nearest = (vr, vc)
            if nearest:
                nr, nc = nearest
                for cr, cc in [(r, nc), (nr, c)]:
                    output[cr][cc] = 2
                dfs(r, c, visited)

def balance_pattern(output: List[List[int]], total_red: int):
    # Ensure no row or column is entirely red or black
    for r in range(5):
        if all(cell == 2 for cell in output[r]) or all(cell == 0 for cell in output[r]):
            c = random.randint(0, 3)
            output[r][c] = 2 if output[r][c] == 0 else 0
    
    for c in range(4):
        if all(output[r][c] == 2 for r in range(5)) or all(output[r][c] == 0 for r in range(5)):
            r = random.randint(0, 4)
            output[r][c] = 2 if output[r][c] == 0 else 0
    
    # Adjust to match total_red
    current_red = sum(sum(row) for row in output) // 2
    while current_red != total_red:
        r, c = random.randint(0, 4), random.randint(0, 3)
        if current_red < total_red and output[r][c] == 0:
            output[r][c] = 2
            current_red += 1
        elif current_red > total_red and output[r][c] == 2:
            output[r][c] = 0
            current_red -= 1
