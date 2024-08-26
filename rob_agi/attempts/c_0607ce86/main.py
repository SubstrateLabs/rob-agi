from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_0607ce86(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying and regularizing patterns in three vertical sections.
    
    1. Identifies three vertical sections separated by black columns.
    2. Analyzes the pattern structure in each section (top part and main pattern).
    3. Creates a perfect pattern by choosing the most common color for each position.
    4. Generates an output grid with the perfect pattern repeated three times vertically in each section.
    5. Ensures consistent spacing between sections and pattern repetitions.
    6. Cleans up the grid by setting all areas outside the main pattern to black (0).
    
    Returns a new grid with regularized and aligned patterns across all three sections.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    sections = find_vertical_sections(input_grid)
    perfect_pattern = create_perfect_pattern(input_grid, sections)
    
    for section_start, section_end in sections:
        repeat_pattern(output_grid, perfect_pattern, section_start, section_end)
    
    return output_grid

def find_vertical_sections(grid: ColoredGrid) -> List[Tuple[int, int]]:
    """Finds the three vertical sections in the grid."""
    _, cols = grid.get_dimensions()
    black_cols = [col for col in range(cols) if all(grid.get_cell(row, col) == 0 for row in range(grid.num_rows))]
    sections = []
    start = 0
    for col in black_cols:
        if col - start > 1:
            sections.append((start, col))
            start = col + 1
    if start < cols:
        sections.append((start, cols))
    return sections[:3]  # Ensure we only return three sections

def create_perfect_pattern(grid: ColoredGrid, sections: List[Tuple[int, int]]) -> ColoredGrid:
    """Creates a perfect pattern based on the most common colors across all sections."""
    pattern_height = find_pattern_height(grid, sections[0])
    pattern_width = sections[0][1] - sections[0][0]
    perfect_pattern = ColoredGrid(values=[[0 for _ in range(pattern_width)] for _ in range(pattern_height)])
    
    for row in range(pattern_height):
        for col in range(pattern_width):
            colors = [grid.get_cell(row, section[0] + col) for section in sections]
            most_common_color = max(set(colors), key=colors.count)
            perfect_pattern.set_cell(row, col, most_common_color)
    
    return perfect_pattern

def find_pattern_height(grid: ColoredGrid, section: Tuple[int, int]) -> int:
    """Determines the height of one complete pattern in a section."""
    start, end = section
    for row in range(1, grid.num_rows // 3):
        if all(grid.get_cell(row, col) == 0 for col in range(start, end)):
            return row
    return grid.num_rows // 3  # Fallback if no clear separator is found

def repeat_pattern(output_grid: ColoredGrid, pattern: ColoredGrid, start: int, end: int):
    """Repeats the pattern three times vertically in the section."""
    pattern_height = pattern.num_rows
    for i in range(3):
        top = i * (pattern_height + 1)  # +1 for spacing
        for row in range(pattern_height):
            for col in range(start, end):
                output_grid.set_cell(top + row, col, pattern.get_cell(row, col - start))
