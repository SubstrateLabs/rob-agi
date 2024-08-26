from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_0607ce86(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying and regularizing patterns in three vertical sections.
    
    1. Identifies three vertical sections separated by black columns.
    2. Analyzes each section to determine the repeating pattern and its height.
    3. Creates a perfect pattern for each section by choosing the most common color for each position.
    4. Generates an output grid with the perfect pattern repeated three times vertically in each section.
    5. Ensures consistent spacing between sections and pattern repetitions.
    6. Cleans up the grid by setting all areas outside the main patterns to black (0).
    
    Returns a new grid with regularized and aligned patterns across all three sections.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    sections = find_vertical_sections(input_grid)
    
    for section_start, section_end in sections:
        section_pattern = create_section_pattern(input_grid, section_start, section_end)
        repeat_pattern(output_grid, section_pattern, section_start, section_end)
    
    # Clean up edges
    for row in range(rows):
        for col in range(sections[-1][1], cols):
            output_grid.set_cell(row, col, 0)
    
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

def create_section_pattern(grid: ColoredGrid, start: int, end: int) -> ColoredGrid:
    """Creates a perfect pattern for a section based on the most common colors."""
    pattern_height = find_pattern_height(grid, start, end)
    pattern_width = end - start
    pattern = ColoredGrid(values=[[0 for _ in range(pattern_width)] for _ in range(pattern_height)])
    
    for row in range(pattern_height):
        for col in range(pattern_width):
            colors = [grid.get_cell(r, start + col) for r in range(row, grid.num_rows, pattern_height)]
            most_common_color = max(set(colors) - {0}, key=colors.count, default=0)
            pattern.set_cell(row, col, most_common_color)
    
    return pattern

def find_pattern_height(grid: ColoredGrid, start: int, end: int) -> int:
    """Determines the height of one complete pattern in a section."""
    for row in range(1, grid.num_rows // 3):
        if all(grid.get_cell(row, col) == 0 for col in range(start, end)):
            return row + 1  # Include the black row in the pattern
    return grid.num_rows // 3  # Fallback if no clear separator is found

def repeat_pattern(output_grid: ColoredGrid, pattern: ColoredGrid, start: int, end: int):
    """Repeats the pattern three times vertically in the section."""
    pattern_height = pattern.num_rows
    available_height = output_grid.num_rows - 1  # Leave the top row black
    
    for i in range(3):
        top = 1 + i * (pattern_height + 1)  # Start from row 1 and add spacing
        for row in range(pattern_height):
            if top + row >= available_height:
                break
            for col in range(start, end):
                output_grid.set_cell(top + row, col, pattern.get_cell(row, col - start))
