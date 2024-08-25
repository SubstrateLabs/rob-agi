from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_66f2d22f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by analyzing 4x2 sections and marking columns of interest.
    
    The function works as follows:
    1. Creates a new 4x7 output grid, initially filled with black (0).
    2. Analyzes each 4x2 section of the input grid (corresponding to 1 column in the output).
    3. If the section contains a continuous line or shape of the same color (green or red)
       that spans at least 3 squares in any direction (horizontal, vertical, or diagonal),
       marks the corresponding column in the output grid as gray (5).
    4. Otherwise, leaves the corresponding column in the output grid as black (0).
    
    This process effectively detects and highlights the presence of significant
    continuous colored regions in the input grid, simplifying the complex color patterns
    into a binary representation.
    """
    output_grid = ColoredGrid(values=[[0 for _ in range(7)] for _ in range(4)])
    
    def check_continuous_shape(section: List[List[int]]) -> bool:
        directions = [(0,1), (1,0), (1,1), (-1,1)]  # right, down, diagonal down-right, diagonal up-right
        for r in range(4):
            for c in range(2):
                if section[r][c] in [2, 3]:  # Check only for red (2) and green (3)
                    color = section[r][c]
                    for dr, dc in directions:
                        count = 0
                        nr, nc = r, c
                        while 0 <= nr < 4 and 0 <= nc < 2 and section[nr][nc] == color:
                            count += 1
                            nr, nc = nr + dr, nc + dc
                        if count >= 3:
                            return True
        return False

    for i in range(7):
        section = [row[2*i:2*i+2] for row in input_grid.values]
        if check_continuous_shape(section):
            for row in range(4):
                output_grid.values[row][i] = 5
    
    return output_grid
