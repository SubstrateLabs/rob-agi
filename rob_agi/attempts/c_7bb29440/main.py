from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_7bb29440(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 7bb29440 challenge by identifying the largest blue rectangle
    containing at least one special square (yellow or magenta) and having
    the most top-left position.

    The function performs the following steps:
    1. Identify all blue regions in the input grid.
    2. For each region, find candidate rectangles containing special squares.
    3. Select the best rectangle based on size and position criteria.
    4. Extract and return the selected rectangle as a new ColoredGrid.

    Args:
    input_grid (ColoredGrid): The input grid to process.

    Returns:
    ColoredGrid: The extracted rectangle as a new ColoredGrid object.
    """
    rows, cols = input_grid.get_dimensions()
    blue_regions = input_grid.find_connected_regions(1)  # Find all blue regions

    candidate_rectangles = []
    for region in blue_regions:
        special_squares = [pos for pos in region if input_grid.get_cell(*pos) in [4, 6]]
        if special_squares:
            for top_left in region:
                for bottom_right in region:
                    if top_left[0] <= bottom_right[0] and top_left[1] <= bottom_right[1]:
                        rect = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
                        if any(top_left[0] <= s[0] <= bottom_right[0] and top_left[1] <= s[1] <= bottom_right[1] for s in special_squares):
                            candidate_rectangles.append(rect)

    if not candidate_rectangles:
        return None

    # Select the best rectangle based on size and position
    best_rect = max(candidate_rectangles, key=lambda r: ((r[2]-r[0]+1)*(r[3]-r[1]+1), -r[0], -r[1]))

    # Extract the selected rectangle
    result = input_grid.extract_subgrid(best_rect[0], best_rect[1], best_rect[2]-best_rect[0]+1, best_rect[3]-best_rect[1]+1)
    return result
