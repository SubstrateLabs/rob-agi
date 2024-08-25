from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_8597cfd7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by following these steps:
    1. Identify the horizontal gray line and vertical lines spanning at least 4 squares.
    2. Count squares below the gray line and total squares for each vertical line.
    3. Select the line with the most squares below the gray line.
    4. In case of a tie, choose the line with the highest total count.
    5. If still tied, select the rightmost line.
    6. Return a 2x2 grid filled with the color of the selected line.
    """
    rows, cols = input_grid.get_dimensions()
    gray_row = next(r for r in range(rows) if input_grid.values[r][0] == 5)
    
    def analyze_line(line: Tuple[int, List[Tuple[int, int]]]) -> Tuple[int, int, int, int]:
        color, coords = line
        below_gray = sum(1 for _, y in coords if y > gray_row)
        total = len(coords)
        rightmost = max(x for x, _ in coords)
        return color, below_gray, total, rightmost
    
    vertical_lines = [line for line in input_grid.detect_lines() if len(line[1]) >= 4]
    analyzed_lines = [analyze_line(line) for line in vertical_lines]
    
    if not analyzed_lines:
        return ColoredGrid(values=[[0, 0], [0, 0]])
    
    selected_color = max(analyzed_lines, key=lambda x: (x[1], x[2], x[3]))[0]
    return ColoredGrid(values=[[selected_color] * 2 for _ in range(2)])
