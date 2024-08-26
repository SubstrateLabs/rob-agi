from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_66f2d22f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input 4x14 grid into a 4x7 output grid based on the following process:
    1. Divides the input grid into seven 4x2 sections.
    2. Analyzes each section for:
       a. Presence of colored cells (2 or 3).
       b. Vertical lines of colored cells.
       c. Horizontal lines of colored cells.
       d. Diagonal patterns of colored cells.
    3. Generates the output based on the analysis:
       a. Marks cells gray (5) if they correspond to significant patterns of colored cells.
       b. Preserves some empty space to maintain the overall structure.
    4. Post-processes the output to ensure connectivity and remove isolated cells.
    5. Returns the resulting 4x7 grid with black (0) and gray (5) cells.

    This process captures essential patterns while maintaining a balance between
    filled and empty spaces in the transformation.
    """
    output_grid = ColoredGrid(values=[[0 for _ in range(7)] for _ in range(4)])
    
    def analyze_section(section: List[List[int]]) -> Tuple[bool, List[int], List[int], bool]:
        has_color = any(cell in [2, 3] for row in section for cell in row)
        vertical_lines = [sum(section[r][c] in [2, 3] for r in range(4)) for c in range(2)]
        horizontal_lines = [sum(cell in [2, 3] for cell in row) for row in section]
        has_diagonal = (section[0][0] in [2, 3] and section[1][1] in [2, 3]) or (section[0][1] in [2, 3] and section[1][0] in [2, 3])
        return has_color, vertical_lines, horizontal_lines, has_diagonal

    for i in range(7):
        section = [row[2*i:2*i+2] for row in input_grid.values]
        has_color, vertical_lines, horizontal_lines, has_diagonal = analyze_section(section)
        
        if has_color:
            # Mark vertical lines
            if any(line >= 3 for line in vertical_lines):
                for r in range(4):
                    if section[r][vertical_lines.index(max(vertical_lines))] in [2, 3]:
                        output_grid.values[r][i] = 5
            
            # Mark horizontal lines
            for r in range(4):
                if horizontal_lines[r] == 2:
                    output_grid.values[r][i] = 5
            
            # Mark diagonal patterns
            if has_diagonal:
                if section[0][0] in [2, 3] and section[1][1] in [2, 3]:
                    output_grid.values[0][i] = 5
                    output_grid.values[1][i] = 5
                elif section[0][1] in [2, 3] and section[1][0] in [2, 3]:
                    output_grid.values[0][i] = 5
                    output_grid.values[1][i] = 5
            
            # Mark bottom cells for density
            if sum(horizontal_lines) >= 3:
                output_grid.values[3][i] = 5

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
                                    for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)] 
                                    if 0 <= nr < 4 and 0 <= nc < 7)
                    if neighbors >= 2:
                        output_grid.values[r][c] = 5

    return output_grid
