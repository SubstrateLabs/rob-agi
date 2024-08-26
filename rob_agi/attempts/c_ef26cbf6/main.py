from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_ef26cbf6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Identifies yellow (4) lines that divide the grid into sections.
    2. Processes the top half:
       - For each section, finds the highest non-yellow, non-zero color.
    3. Processes the bottom half:
       - For each section, replaces non-yellow cells with the color from the corresponding top section.
    4. Preserves yellow lines and originally empty cells not part of colored regions.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = input_grid.deep_copy()
    yellow_lines = grid.detect_lines()
    sections = get_sections(grid, yellow_lines)
    main_horizontal = find_main_horizontal(yellow_lines)
    
    top_colors = process_top_half(grid, sections, main_horizontal)
    process_bottom_half(grid, sections, main_horizontal, top_colors)
    
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

def find_main_horizontal(yellow_lines: List[Tuple[int, List[Tuple[int, int]]]]) -> int:
    horizontal_lines = [line[1][0][0] for line in yellow_lines if len(line[1]) > 1 and line[1][0][0] == line[1][1][0]]
    return max(horizontal_lines) if horizontal_lines else 0

def process_top_half(grid: ColoredGrid, sections: List[Tuple[int, int, int, int]], main_horizontal: int) -> Dict[int, int]:
    top_colors = {}
    for i, section in enumerate(sections):
        if section[2] <= main_horizontal:
            max_color = find_max_color(grid, section)
            if max_color > 0:
                top_colors[i] = max_color
    return top_colors

def process_bottom_half(grid: ColoredGrid, sections: List[Tuple[int, int, int, int]], main_horizontal: int, top_colors: Dict[int, int]):
    sections_per_row = len([s for s in sections if s[2] <= main_horizontal])
    for i, section in enumerate(sections):
        if section[0] > main_horizontal:
            top_section_index = i % sections_per_row
            if top_section_index in top_colors:
                color = top_colors[top_section_index]
                fill_section(grid, section, color)

def find_max_color(grid: ColoredGrid, section: Tuple[int, int, int, int]) -> int:
    top, left, bottom, right = section
    return max((grid.values[r][c] for r in range(top, bottom) for c in range(left, right) if grid.values[r][c] not in [0, 4]), default=0)

def fill_section(grid: ColoredGrid, section: Tuple[int, int, int, int], color: int):
    top, left, bottom, right = section
    for r in range(top, bottom):
        for c in range(left, right):
            if grid.values[r][c] != 4:  # Preserve yellow lines
                if grid.values[r][c] != 0 or any(grid.values[nr][nc] != 0 for nr, nc in get_neighbors(r, c, top, left, bottom, right)):
                    grid.values[r][c] = color

def get_neighbors(r: int, c: int, top: int, left: int, bottom: int, right: int) -> List[Tuple[int, int]]:
    return [(nr, nc) for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            if top <= nr < bottom and left <= nc < right]
