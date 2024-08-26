from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_66f2d22f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input 4x14 grid into a 4x7 output grid based on the following process:
    1. Divides the input grid into seven 4x2 sections.
    2. Creates an influence map to capture the effect of colored cells.
    3. Analyzes each section for:
       a. Total colored cells (2 or 3).
       b. Vertical lines of colored cells.
       c. Rows with two colored cells.
    4. Generates the output based on the analysis and influence map:
       a. Marks entire columns gray if the section has 3+ colored cells.
       b. Marks cells gray based on vertical lines, row patterns, and influence.
    5. Post-processes the output to ensure connectivity and remove isolated cells.
    6. Returns the resulting 4x7 grid with black (0) and gray (5) cells.

    This process captures complex patterns, preserves connectivity, and considers
    both local and global influences in the transformation.
    """
    output_grid = ColoredGrid(values=[[0 for _ in range(7)] for _ in range(4)])
    influence_map = [[0 for _ in range(7)] for _ in range(4)]
    
    def create_influence_map(grid: List[List[int]]) -> None:
        for r in range(4):
            for c in range(14):
                if grid[r][c] in [2, 3]:
                    section = c // 2
                    influence_map[r][section] += 2
                    for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                        nr, nc = r + dr, section + dc
                        if 0 <= nr < 4 and 0 <= nc < 7:
                            influence_map[nr][nc] += 1

    def analyze_section(section: List[List[int]]) -> Tuple[int, List[int], List[int]]:
        total_colored = sum(cell in [2, 3] for row in section for cell in row)
        vertical_lines = [sum(section[r][c] in [2, 3] for r in range(4)) for c in range(2)]
        row_patterns = [sum(cell in [2, 3] for cell in row) for row in section]
        return total_colored, vertical_lines, row_patterns

    create_influence_map(input_grid.values)

    for i in range(7):
        section = [row[2*i:2*i+2] for row in input_grid.values]
        total_colored, vertical_lines, row_patterns = analyze_section(section)
        
        if total_colored >= 3:
            for r in range(4):
                output_grid.values[r][i] = 5
        else:
            for r in range(4):
                if vertical_lines[0] >= 3 or vertical_lines[1] >= 3 or row_patterns[r] == 2 or influence_map[r][i] >= 3:
                    output_grid.values[r][i] = 5

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
