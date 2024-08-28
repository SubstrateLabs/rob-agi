from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_66f2d22f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input 4x14 grid into a 4x7 output grid based on the following process:
    1. Divides the input grid into seven 4x2 sections.
    2. Analyzes each section for colored regions (2 and 3) and their patterns.
    3. Creates an abstract representation using gray (5) cells in the output.
    4. Captures both local and global patterns across the input grid.
    5. Balances the representation by maintaining appropriate density.
    6. Refines the output to ensure continuity and coherent shapes.
    7. Returns the resulting 4x7 grid with black (0) and gray (5) cells.

    This process aims to create a meaningful abstraction that reflects the 
    essential characteristics and patterns of the input grid.
    """
    output_grid = ColoredGrid(values=[[0 for _ in range(7)] for _ in range(4)])
    
    def analyze_section(section: List[List[int]]) -> Tuple[float, List[Tuple[int, int]]]:
        colored_cells = [(r, c) for r in range(4) for c in range(2) if section[r][c] in [2, 3]]
        density = len(colored_cells) / 8
        return density, colored_cells

    sections = [analyze_section([row[2*i:2*i+2] for row in input_grid.values]) for i in range(7)]
    total_density = sum(density for density, _ in sections) / 7

    # Initial fill based on density and patterns
    for i, (density, colored_cells) in enumerate(sections):
        if density > total_density:
            for r, c in colored_cells:
                output_grid.values[r][i] = 5

    # Global pattern analysis and refinement
    for r in range(4):
        for c in range(7):
            if c > 0 and c < 6:
                if output_grid.values[r][c-1] == 5 and output_grid.values[r][c+1] == 5:
                    output_grid.values[r][c] = 5
            if r > 0 and r < 3:
                if output_grid.values[r-1][c] == 5 and output_grid.values[r+1][c] == 5:
                    output_grid.values[r][c] = 5

    # Balance check and adjustment
    output_density = sum(sum(row) for row in output_grid.values) / (4 * 7 * 5)
    if output_density < total_density * 0.8:
        for r in range(4):
            for c in range(7):
                if output_grid.values[r][c] == 0:
                    neighbors = sum(output_grid.values[nr][nc] == 5 
                                    for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)] 
                                    if 0 <= nr < 4 and 0 <= nc < 7)
                    if neighbors >= 2:
                        output_grid.values[r][c] = 5
    elif output_density > total_density * 1.2:
        for r in range(4):
            for c in range(7):
                if output_grid.values[r][c] == 5:
                    neighbors = sum(output_grid.values[nr][nc] == 5 
                                    for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)] 
                                    if 0 <= nr < 4 and 0 <= nc < 7)
                    if neighbors <= 1:
                        output_grid.values[r][c] = 0

    return output_grid
