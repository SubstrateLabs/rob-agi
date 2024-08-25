from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9b365c51(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving vertical color lines from the left side
    to fill a sky blue (8) region on the right side of the grid.

    1. Identifies unique vertical color lines on the left side of the grid.
    2. Determines the starting color for filling.
    3. Clears the left side of the grid.
    4. Identifies the sky blue region on the right side.
    5. Analyzes the structure of the sky blue region.
    6. Assigns colors to horizontal sections of the sky blue region.
    7. Fills the sky blue region with assigned colors.
    8. Preserves non-sky blue cells.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    # Step 1: Identify unique vertical color lines
    colors = identify_vertical_lines(input_grid)

    # Step 2: Determine the starting color
    start_color = determine_start_color(colors)

    # Step 3: Clear the left side of the grid
    output_grid = clear_left_side(input_grid)

    # Step 4: Identify the sky blue region
    sky_blue_region = identify_sky_blue_region(output_grid)

    # Step 5: Analyze the sky blue region structure
    sections = analyze_sky_blue_structure(sky_blue_region)

    # Step 6 & 7: Assign colors and fill the sky blue region
    fill_sky_blue_region(output_grid, sky_blue_region, sections, colors, start_color)

    return output_grid

def identify_vertical_lines(grid: ColoredGrid) -> List[int]:
    colors = []
    for col in range(7):  # Assume vertical lines are within first 7 columns
        color = next((cell for cell in grid.values[col] if cell != 0), None)
        if color and color not in colors:
            colors.append(color)
    return colors

def determine_start_color(colors: List[int]) -> int:
    if len(colors) > 1:
        return colors[1]  # Start with the second color if available
    return colors[0] if colors else 1  # Default to blue if no colors found

def clear_left_side(grid: ColoredGrid) -> ColoredGrid:
    new_grid = grid.deep_copy()
    for row in new_grid.values:
        for col in range(7):  # Clear first 7 columns
            row[col] = 0
    return new_grid

def identify_sky_blue_region(grid: ColoredGrid) -> List[Tuple[int, int]]:
    sky_blue_region = []
    for r, row in enumerate(grid.values):
        for c, cell in enumerate(row):
            if cell == 8:
                sky_blue_region.append((r, c))
    return sky_blue_region

def analyze_sky_blue_structure(region: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    if not region:
        return []
    
    region.sort()  # Sort by row, then column
    sections = []
    current_section = []
    prev_row = region[0][0]

    for r, c in region:
        if r != prev_row:
            if current_section:
                sections.append(current_section)
                current_section = []
            prev_row = r
        current_section.append((r, c))

    if current_section:
        sections.append(current_section)

    return sections

def fill_sky_blue_region(grid: ColoredGrid, region: List[Tuple[int, int]], sections: List[List[Tuple[int, int]]], colors: List[int], start_color: int):
    color_index = colors.index(start_color)
    for section in sections:
        color = colors[color_index % len(colors)]
        for r, c in section:
            grid.values[r][c] = color
        color_index += 1
