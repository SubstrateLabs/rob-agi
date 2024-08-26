from rob_agi.colored_grid import ColoredGrid
from typing import List

def solve_66f2d22f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input 4x14 grid into a 4x7 output grid based on the following rules:
    1. Divides the input grid into seven 4x2 sections.
    2. For each section:
       a. If a row has two colored cells (2 or 3), mark the corresponding cell in the output column as gray (5).
       b. If there's a vertical line of three or more colored cells, mark the corresponding cells in the output column as gray (5).
       c. If the total number of colored cells in the section is 5 or more, mark the entire output column as gray (5).
    3. Post-process the output grid to connect adjacent gray cells and fill small gaps.
    4. Returns the resulting 4x7 grid with black (0) and gray (5) cells.

    This process simplifies complex color patterns into a binary representation,
    highlighting significant colored regions and vertical patterns while maintaining some structure.
    """
    output_grid = ColoredGrid(values=[[0 for _ in range(7)] for _ in range(4)])
    
    def analyze_section(section: List[List[int]]) -> List[int]:
        result = [0] * 4
        total_colored = sum(cell in [2, 3] for row in section for cell in row)
        
        # Check rows for two colored cells
        for r in range(4):
            if sum(1 for cell in section[r] if cell in [2, 3]) == 2:
                result[r] = 5
        
        # Check for vertical line of three or more colored cells
        for c in range(2):
            colored_cells = [r for r in range(4) if section[r][c] in [2, 3]]
            if len(colored_cells) >= 3:
                for r in colored_cells:
                    result[r] = 5
        
        # Check if total colored cells is 5 or more
        if total_colored >= 5:
            return [5] * 4
        
        return result

    for i in range(7):
        section = [row[2*i:2*i+2] for row in input_grid.values]
        column_result = analyze_section(section)
        for row in range(4):
            output_grid.values[row][i] = column_result[row]
    
    # Post-processing: connect adjacent gray cells and fill small gaps
    for row in range(4):
        for col in range(1, 6):
            if output_grid.values[row][col-1] == 5 and output_grid.values[row][col+1] == 5:
                output_grid.values[row][col] = 5
    
    return output_grid
