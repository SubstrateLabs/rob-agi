from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_66f2d22f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input 4x14 grid into a 4x7 output grid based on the following process:
    1. Divides the input grid into seven 4x2 sections.
    2. Analyzes each section for significant colored regions (2 or 3).
    3. Creates an abstract outline of these regions using gray (5) cells in the output.
    4. Balances the representation by maintaining a good ratio of filled to empty spaces.
    5. Refines the output to ensure continuity where appropriate and remove isolated cells.
    6. Returns the resulting 4x7 grid with black (0) and gray (5) cells.

    This process captures the essence of the input patterns while maintaining a
    balance between detail and abstraction in the transformation.
    """
    output_grid = ColoredGrid(values=[[0 for _ in range(7)] for _ in range(4)])
    
    def analyze_section(section: List[List[int]]) -> List[int]:
        color_density = [sum(1 for r in range(4) if section[r][c] in [2, 3]) for c in range(2)]
        return [sum(color_density) / 2] * 2  # Average density for the column

    total_density = 0
    densities = []

    for i in range(7):
        section = [row[2*i:2*i+2] for row in input_grid.values]
        section_density = analyze_section(section)
        densities.extend(section_density)
        total_density += sum(section_density)

    avg_density = total_density / 14
    threshold = avg_density * 1.2  # Adjust this factor to balance filled vs empty space

    for i, density in enumerate(densities):
        if density >= threshold:
            col = i // 2
            if density > threshold * 1.5:  # High density
                for r in range(4):
                    output_grid.values[r][col] = 5
            elif density > threshold * 1.2:  # Medium-high density
                for r in range(3):
                    output_grid.values[r][col] = 5
            else:  # Medium density
                for r in range(2):
                    output_grid.values[r][col] = 5

    # Refine the output
    for r in range(4):
        for c in range(7):
            if output_grid.values[r][c] == 5:
                neighbors = sum(output_grid.values[nr][nc] == 5 
                                for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)] 
                                if 0 <= nr < 4 and 0 <= nc < 7)
                if neighbors == 0:
                    output_grid.values[r][c] = 0
            elif c > 0 and c < 6:
                if output_grid.values[r][c-1] == 5 and output_grid.values[r][c+1] == 5:
                    output_grid.values[r][c] = 5

    return output_grid
