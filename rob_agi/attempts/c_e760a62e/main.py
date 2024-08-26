from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e760a62e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored squares according to specific rules:
    1. Identifies section boundaries defined by sky blue (8) lines.
    2. Expands green (3) squares vertically across sections and horizontally within sections.
    3. Creates magenta (6) squares above original green squares within the same section.
    4. Expands red (2) squares horizontally within their section and vertically if adjacent to expanded green areas.
    5. Fills empty cells to the left of expanded green areas with red (2) within the same section.
    6. Processes colors in order: green, then red, then fill remaining.
    7. Respects sky blue (8) grid lines as boundaries.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the expansion rules.
    """
    output_grid = input_grid.deep_copy()
    sections = find_sections(output_grid)
    
    # Process green squares
    for y in range(output_grid.num_rows):
        for x in range(output_grid.num_cols):
            if output_grid.values[y][x] == 3:
                expand_green(output_grid, sections, x, y)
    
    # Process red squares
    for y in range(output_grid.num_rows):
        for x in range(output_grid.num_cols):
            if input_grid.values[y][x] == 2:  # Check original grid for red
                expand_red(output_grid, sections, x, y)
    
    # Fill remaining cells to the left of green areas
    fill_left_of_green(output_grid, sections)
    
    return output_grid

def find_sections(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    """Identifies and returns section boundaries defined by sky blue (8) lines."""
    rows, cols = grid.get_dimensions()
    sections = []
    start_row, start_col = 0, 0

    for r in range(rows):
        if all(grid.values[r][c] == 8 for c in range(cols)):
            if start_row < r:
                for c in range(cols):
                    if grid.values[start_row][c] == 8:
                        if start_col < c:
                            sections.append((start_row, start_col, r - 1, c - 1))
                        start_col = c + 1
            start_row = r + 1
            start_col = 0

    return sections

def expand_green(grid: ColoredGrid, sections: List[Tuple[int, int, int, int]], x: int, y: int):
    """Handles green expansion and magenta creation."""
    section = next((s for s in sections if s[0] <= y <= s[2] and s[1] <= x <= s[3]), None)
    if section:
        top, left, bottom, right = section
        # Vertical expansion
        for r in range(grid.num_rows):
            if grid.values[r][x] in [0, 3]:
                grid.values[r][x] = 3
        # Horizontal expansion within section
        for c in range(left, right + 1):
            if grid.values[y][c] in [0, 3]:
                grid.values[y][c] = 3
        # Create magenta above
        if y > top and grid.values[y-1][x] == 0:
            grid.values[y-1][x] = 6

def expand_red(grid: ColoredGrid, sections: List[Tuple[int, int, int, int]], x: int, y: int):
    """Handles red expansion."""
    section = next((s for s in sections if s[0] <= y <= s[2] and s[1] <= x <= s[3]), None)
    if section:
        top, left, bottom, right = section
        # Horizontal expansion within section
        for c in range(left, right + 1):
            if grid.values[y][c] == 0:
                grid.values[y][c] = 2
        # Vertical expansion if adjacent to green
        if (x > 0 and grid.values[y][x-1] == 3) or (x < grid.num_cols - 1 and grid.values[y][x+1] == 3):
            for r in range(top, bottom + 1):
                if grid.values[r][x] == 0:
                    grid.values[r][x] = 2

def fill_left_of_green(grid: ColoredGrid, sections: List[Tuple[int, int, int, int]]):
    """Fills empty cells to the left of green areas with red."""
    for section in sections:
        top, left, bottom, right = section
        for y in range(top, bottom + 1):
            green_found = False
            for x in range(left, right + 1):
                if grid.values[y][x] == 3:
                    green_found = True
                elif green_found and grid.values[y][x] == 0:
                    grid.values[y][x] = 2
