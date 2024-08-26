from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_34b99a2b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the 5x9 input grid into a 5x4 output grid based on the following steps:
    1. Splits the input into left and right halves, analyzing sky (8) and gray (5) regions.
    2. Calculates activity scores for each quadrant of the input.
    3. Maps activity scores to the output grid, placing red (2) squares accordingly.
    4. Applies smoothing rules to connect nearby red cells.
    5. Ensures vertical consistency in columns with multiple red cells.
    6. Processes the bottom row based on input density.
    7. Makes final adjustments to remove isolated cells and balance the pattern.
    """
    left_half, right_half = split_grid(input_grid)
    output = [[0 for _ in range(4)] for _ in range(5)]
    
    activity_scores = calculate_activity_scores(left_half, right_half)
    map_activity_to_output(activity_scores, output)
    
    apply_smoothing_rules(output)
    ensure_vertical_consistency(output)
    process_bottom_row(left_half, right_half, output)
    remove_isolated_cells(output)
    final_adjustments(output)
    
    return ColoredGrid(values=output)

def split_grid(grid: ColoredGrid) -> Tuple[List[List[int]], List[List[int]]]:
    left_half = [row[:4] for row in grid.values]
    right_half = [row[5:] for row in grid.values]
    return left_half, right_half

def calculate_activity_scores(left_half: List[List[int]], right_half: List[List[int]]) -> List[float]:
    scores = []
    for half, color in [(left_half, 8), (right_half, 5)]:
        for rows in [slice(0, 3), slice(3, 5)]:
            quadrant = [row[rows] for row in half]
            score = (
                sum(cell == color for row in quadrant for cell in row) / len(quadrant) / len(quadrant[0]) +
                len(identify_vertical_lines(quadrant, color)) / len(quadrant[0]) +
                len(identify_diagonal_lines(quadrant, color)) / (len(quadrant) * len(quadrant[0]))
            ) / 3
            scores.append(score)
    return scores

def map_activity_to_output(scores: List[float], output: List[List[int]]):
    quadrants = [(0, 0), (3, 0), (0, 2), (3, 2)]
    for score, (start_row, start_col) in zip(scores, quadrants):
        num_red = int(score * 4)
        for _ in range(num_red):
            r, c = start_row + _ // 2, start_col + _ % 2
            output[r][c] = 2

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
