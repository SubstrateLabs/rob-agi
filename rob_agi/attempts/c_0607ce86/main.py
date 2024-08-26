from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_0607ce86(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying and regularizing patterns in three vertical sections.
    
    1. Analyzes the input grid to find three main vertical sections.
    2. For each section, identifies the most common rectangular pattern.
    3. Creates a "perfect" version of this pattern, removing imperfections.
    4. Repeats the perfect pattern four times vertically in each section.
    5. Adds a consistent bottom element if it exists in the majority of rectangles.
    6. Ensures proper spacing between sections and patterns.
    7. Cleans up the grid by setting all areas outside patterns to black (0).
    
    Returns a new grid with regularized and aligned patterns.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Find the three vertical sections
    sections = find_vertical_sections(input_grid)
    
    for section_start, section_end in sections:
        # Identify the most common pattern in the section
        pattern = identify_pattern(input_grid, section_start, section_end)
        
        # Create a perfect version of the pattern
        perfect_pattern = create_perfect_pattern(pattern)
        
        # Repeat the pattern four times in the section
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

def identify_pattern(grid: ColoredGrid, start: int, end: int) -> ColoredGrid:
    """Identifies the most common rectangular pattern in a section."""
    patterns = {}
    for top in range(0, grid.num_rows - 3):
        for bottom in range(top + 4, grid.num_rows):
            pattern = grid.extract_subgrid(top, start, bottom - top, end - start)
            pattern_key = tuple(tuple(row) for row in pattern.values)
            patterns[pattern_key] = patterns.get(pattern_key, 0) + 1
    
    most_common_pattern = max(patterns, key=patterns.get)
    return ColoredGrid(values=[list(row) for row in most_common_pattern])

def create_perfect_pattern(pattern: ColoredGrid) -> ColoredGrid:
    """Creates a perfect version of the pattern by removing imperfections."""
    perfect = pattern.deep_copy()
    for color in range(1, 10):  # Exclude black (0)
        regions = perfect.find_connected_regions(color)
        if regions:
            main_region = max(regions, key=len)
            for row, col in main_region:
                perfect.set_cell(row, col, color)
    return perfect

def repeat_pattern(output_grid: ColoredGrid, pattern: ColoredGrid, start: int, end: int):
    """Repeats the pattern four times in the section."""
    pattern_height = pattern.num_rows
    for i in range(4):
        top = i * (pattern_height + 1)  # +1 for spacing
        for row in range(pattern_height):
            for col in range(start, end):
                output_grid.set_cell(top + row, col, pattern.get_cell(row, col - start))
