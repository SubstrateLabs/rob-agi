from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_0607ce86(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying and regularizing patterns in three vertical sections.
    
    1. Identifies three vertical sections separated by black columns.
    2. Extracts a 4-row pattern from the first section, ignoring black rows.
    3. Creates an output grid with the pattern repeated three times in each section.
    4. Ensures consistent black rows between pattern repetitions and sections.
    5. Cleans up the grid by setting all areas outside the main patterns to black (0).
    
    Returns a new grid with regularized and aligned patterns across all three sections.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    sections = find_vertical_sections(input_grid)
    pattern = extract_pattern(input_grid, sections[0][0], sections[0][1])
    
    for section_start, section_end in sections:
        apply_pattern(output_grid, pattern, section_start, section_end)
    
    clean_up_grid(output_grid, sections)
    
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

def extract_pattern(grid: ColoredGrid, start: int, end: int) -> ColoredGrid:
    """Extracts a 4-row pattern from the section, ignoring black rows."""
    pattern_rows = []
    row = 1  # Start from the second row to skip the top black row
    while len(pattern_rows) < 4 and row < grid.num_rows:
        if any(grid.get_cell(row, col) != 0 for col in range(start, end)):
            pattern_rows.append([grid.get_cell(row, col) for col in range(start, end)])
        row += 1
    return ColoredGrid(values=pattern_rows)

def apply_pattern(output_grid: ColoredGrid, pattern: ColoredGrid, start: int, end: int):
    """Applies the pattern three times in the section with proper spacing."""
    pattern_height = pattern.num_rows
    section_width = end - start
    
    for i in range(3):
        top = 1 + i * 5  # Start from row 1 and repeat every 5 rows
        for row in range(pattern_height):
            for col in range(section_width):
                if col < pattern.num_cols:
                    output_grid.set_cell(top + row, start + col, pattern.get_cell(row, col))
                else:
                    output_grid.set_cell(top + row, start + col, pattern.get_cell(row, -1))  # Repeat last column

def clean_up_grid(output_grid: ColoredGrid, sections: List[Tuple[int, int]]):
    """Ensures black rows and columns are in place."""
    rows, cols = output_grid.get_dimensions()
    
    # Set top and bottom rows to black
    for col in range(cols):
        output_grid.set_cell(0, col, 0)
        output_grid.set_cell(rows - 1, col, 0)
    
    # Set separating rows to black
    for row in [5, 10, 15]:
        for col in range(cols):
            output_grid.set_cell(row, col, 0)
    
    # Set separating columns to black
    for section_end, next_section_start in zip([s[1] for s in sections[:-1]], [s[0] for s in sections[1:]]):
        for row in range(rows):
            for col in range(section_end, next_section_start):
                output_grid.set_cell(row, col, 0)
    
    # Clean up right edge
    for row in range(rows):
        for col in range(sections[-1][1], cols):
            output_grid.set_cell(row, col, 0)
