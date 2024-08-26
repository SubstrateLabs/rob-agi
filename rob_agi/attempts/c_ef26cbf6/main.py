from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ef26cbf6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Detects yellow (4) lines that divide the grid into sections.
    2. For each section:
       - Finds the maximum non-yellow, non-zero color value.
       - Fills all non-yellow cells and adjacent empty cells with this color.
    3. Preserves the yellow lines and empty cells not adjacent to colored cells.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = input_grid.deep_copy()
    yellow_lines = grid.detect_lines()
    sections = get_sections(grid, yellow_lines)
    
    for section in sections:
        process_section(grid, section)
    
    return grid

def get_sections(grid: ColoredGrid, yellow_lines: List[Tuple[int, List[Tuple[int, int]]]]) -> List[Tuple[int, int, int, int]]:
    sections = []
    rows, cols = grid.get_dimensions()
    
    h_lines = sorted([line[1][0][0] for line in yellow_lines if len(line[1]) > 1 and line[1][0][0] == line[1][1][0]])
    v_lines = sorted([line[1][0][1] for line in yellow_lines if len(line[1]) > 1 and line[1][0][1] == line[1][1][1]])
    
    h_lines = [-1] + h_lines + [rows]
    v_lines = [-1] + v_lines + [cols]
    
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            top = h_lines[i] + 1
            bottom = h_lines[i+1]
            left = v_lines[j] + 1
            right = v_lines[j+1]
            sections.append((top, left, bottom, right))
    
    return sections

def process_section(grid: ColoredGrid, section_coords: Tuple[int, int, int, int]):
    top, left, bottom, right = section_coords
    max_color = 0
    cells_to_fill = set()
    
    for r in range(top, bottom):
        for c in range(left, right):
            if grid.values[r][c] not in [0, 4]:
                max_color = max(max_color, grid.values[r][c])
                cells_to_fill.add((r, c))
                # Check adjacent cells
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if top <= nr < bottom and left <= nc < right and grid.values[nr][nc] == 0:
                        cells_to_fill.add((nr, nc))
    
    if max_color == 0:
        return
    
    for r, c in cells_to_fill:
        grid.values[r][c] = max_color
