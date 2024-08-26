from rob_agi.colored_grid import ColoredGrid

GRID_SIZE = 7
SECTION_SIZE = 3
OUTPUT_SIZE = 3

def count_non_black_cells(grid: ColoredGrid, top: int, left: int, size: int) -> int:
    count = 0
    for r in range(top, top + size):
        for c in range(left, left + size):
            if grid.get_cell(r, c) != 0:
                count += 1
    return count

def create_output_grid(pattern: str) -> ColoredGrid:
    if pattern == "horizontal_stripe":
        return ColoredGrid(values=[[0,0,0],[8,8,8],[0,0,0]])
    elif pattern == "inverted_l":
        return ColoredGrid(values=[[8,8,8],[8,0,0],[8,0,0]])
    else:  # backwards_c
        return ColoredGrid(values=[[0,8,8],[0,8,0],[0,8,0]])

def solve_9110e3c5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 7x7 input grid into a 3x3 output grid based on the distribution of non-black cells.
    
    The function divides the input grid into nine 3x3 sections and analyzes the distribution
    of non-black cells across these sections. Based on this analysis, it determines one of
    three output patterns:
    1. Horizontal stripe (balanced distribution or high corner concentration)
    2. Inverted "L" (left-heavy distribution)
    3. Backwards "C" (right-heavy distribution)

    Args:
    input_grid (ColoredGrid): A 7x7 ColoredGrid object representing the input.

    Returns:
    ColoredGrid: A 3x3 ColoredGrid object representing the output pattern.
    """
    section_counts = [
        count_non_black_cells(input_grid, r, c, SECTION_SIZE)
        for r in range(0, GRID_SIZE, SECTION_SIZE)
        for c in range(0, GRID_SIZE, SECTION_SIZE)
    ]

    corner_count = section_counts[0] + section_counts[2] + section_counts[6] + section_counts[8]
    side_count = section_counts[1] + section_counts[3] + section_counts[5] + section_counts[7]
    left_count = section_counts[0] + section_counts[3] + section_counts[6]
    right_count = section_counts[2] + section_counts[5] + section_counts[8]

    if corner_count > (side_count * 1.5):
        return create_output_grid("horizontal_stripe")
    elif left_count > (right_count * 1.2):
        return create_output_grid("inverted_l")
    elif right_count > (left_count * 1.2):
        return create_output_grid("backwards_c")
    else:
        return create_output_grid("horizontal_stripe")
