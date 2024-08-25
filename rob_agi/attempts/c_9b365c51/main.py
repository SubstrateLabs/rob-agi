from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9b365c51(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving vertical color lines from the left side
    to fill a sky blue (8) region on the right side of the grid.

    1. Identifies vertical color lines on the left side of the grid.
    2. Clears the left side of the grid.
    3. Identifies the sky blue region on the right side.
    4. Divides the sky blue region into sections based on the number of colors.
    5. Assigns colors to the sections in a specific order.
    6. Creates the output grid by filling the sections with assigned colors.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    # Step 1: Identify vertical color lines
    colors = identify_vertical_lines(input_grid)

    # Step 2: Clear the left side of the grid
    output_grid = clear_left_side(input_grid)

    # Step 3: Identify the sky blue region
    sky_blue_region = identify_sky_blue_region(output_grid)

    # Step 4 & 5: Divide the sky blue region and assign colors
    sections = divide_and_assign_colors(sky_blue_region, colors)

    # Step 6: Fill the sections with assigned colors
    fill_sections(output_grid, sections)

    return output_grid

def identify_vertical_lines(grid: ColoredGrid) -> List[int]:
    colors = []
    for col in range(7):  # Assume vertical lines are within first 7 columns
        color = next((cell for cell in grid.values[0] if cell != 0), None)
        if color and all(row[col] == color for row in grid.values):
            colors.append(color)
    return colors

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

def divide_and_assign_colors(region: List[Tuple[int, int]], colors: List[int]) -> List[Tuple[List[Tuple[int, int]], int]]:
    if not region or not colors:
        return []

    min_r = min(r for r, _ in region)
    max_r = max(r for r, _ in region)
    min_c = min(c for _, c in region)
    max_c = max(c for _, c in region)
    mid_r = (min_r + max_r) // 2
    mid_c = (min_c + max_c) // 2

    sections = []
    if len(colors) == 2:
        sections = [
            ([(r, c) for r, c in region if c < mid_c], colors[0]),
            ([(r, c) for r, c in region if c >= mid_c], colors[1])
        ]
    elif len(colors) == 3:
        sections = [
            ([(r, c) for r, c in region if r < mid_r and c < mid_c], colors[0]),
            ([(r, c) for r, c in region if r >= mid_r and c < mid_c], colors[1]),
            ([(r, c) for r, c in region if c >= mid_c], colors[2])
        ]
    elif len(colors) >= 4:
        sections = [
            ([(r, c) for r, c in region if r < mid_r and c < mid_c], colors[0]),
            ([(r, c) for r, c in region if r >= mid_r and c < mid_c], colors[1]),
            ([(r, c) for r, c in region if r < mid_r and c >= mid_c], colors[2]),
            ([(r, c) for r, c in region if r >= mid_r and c >= mid_c], colors[3])
        ]
        # If there are more colors, assign them to additional sections
        for i, color in enumerate(colors[4:], start=4):
            section = [(r, c) for r, c in region if r >= min_r + i * (max_r - min_r) // len(colors) and r < min_r + (i + 1) * (max_r - min_r) // len(colors)]
            sections.append((section, color))

    return sections

def fill_sections(grid: ColoredGrid, sections: List[Tuple[List[Tuple[int, int]], int]]):
    for section, color in sections:
        for r, c in section:
            grid.values[r][c] = color
