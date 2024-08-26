from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_66f2d22f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input 4x14 grid into a 4x7 output grid based on the following process:
    1. Divides the input grid into seven 4x2 sections.
    2. Analyzes each section for:
       a. Total colored cells (2 or 3).
       b. Vertical lines of colored cells.
       c. Horizontal lines of colored cells.
    3. Generates the output based on the analysis:
       a. Marks cells gray (5) if they correspond to a vertical or horizontal line of colored cells.
       b. Marks additional cells gray based on the density and distribution of colored cells in the section.
    4. Post-processes the output to ensure connectivity and remove isolated cells.
    5. Returns the resulting 4x7 grid with black (0) and gray (5) cells.

    This process captures complex patterns, preserves connectivity, and considers
    both local and global influences in the transformation.
    """
    output_grid = ColoredGrid(values=[[0 for _ in range(7)] for _ in range(4)])
    
    def analyze_section(section: List[List[int]]) -> Tuple[int, List[int], List[int]]:
        total_colored = sum(cell in [2, 3] for row in section for cell in row)
        vertical_lines = [sum(section[r][c] in [2, 3] for r in range(4)) for c in range(2)]
        horizontal_lines = [sum(cell in [2, 3] for cell in row) for row in section]
        return total_colored, vertical_lines, horizontal_lines

    for i in range(7):
        section = [row[2*i:2*i+2] for row in input_grid.values]
        total_colored, vertical_lines, horizontal_lines = analyze_section(section)
        
        # Mark vertical lines
        for c in range(2):
            if vertical_lines[c] >= 3:
                for r in range(4):
                    output_grid.values[r][i] = 5
        
        # Mark horizontal lines
        for r in range(4):
            if horizontal_lines[r] == 2:
                output_grid.values[r][i] = 5
        
        # Mark based on density and distribution
        if total_colored >= 4:
            output_grid.values[3][i] = 5  # Mark bottom cell
        elif total_colored == 3:
            if vertical_lines[0] == 2 and vertical_lines[1] == 1:
                output_grid.values[3][i] = 5  # Mark bottom cell
            elif vertical_lines[1] == 2 and vertical_lines[0] == 1:
                output_grid.values[2][i] = 5  # Mark second from bottom

    # Post-processing: ensure connectivity and remove isolated cells
    for _ in range(2):  # Apply twice to handle complex cases
        for r in range(4):
            for c in range(7):
                if output_grid.values[r][c] == 5:
                    neighbors = sum(output_grid.values[nr][nc] == 5 
                                    for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)] 
                                    if 0 <= nr < 4 and 0 <= nc < 7)
                    if neighbors == 0:
                        output_grid.values[r][c] = 0
                else:
                    neighbors = sum(output_grid.values[nr][nc] == 5 
                                    for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1), (r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)] 
                                    if 0 <= nr < 4 and 0 <= nc < 7)
                    if neighbors >= 3:
                        output_grid.values[r][c] = 5

    return output_grid
