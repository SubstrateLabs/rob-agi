from rob_agi.colored_grid import ColoredGrid

GRID_SIZE = 7
SECTION_SIZE = 3
OUTPUT_SIZE = 3

def count_non_black_cells(grid: ColoredGrid, top: int, left: int, size: int) -> int:
    count = 0
    for r in range(top, min(top + size, GRID_SIZE)):
        for c in range(left, min(left + size, GRID_SIZE)):
            if grid.get_cell(r, c) != 0:
                count += 1
    return count

def create_output_grid(pattern: str) -> ColoredGrid:
    if pattern == "horizontal_stripe":
        return ColoredGrid(values=[[0,0,0],[8,8,8],[0,0,0]])
    else:  # backwards_c
        return ColoredGrid(values=[[0,8,8],[0,8,0],[0,8,0]])

def solve_9110e3c5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 7x7 input grid into a 3x3 output grid based on the distribution of non-black cells.
    
    The function divides the input grid into nine sections and analyzes the distribution
    of non-black cells across these sections. Based on this analysis, it determines one of
    two output patterns:
    1. Horizontal stripe (when the middle row has a high concentration of non-black cells)
    2. Backwards "C" (when the right side has more non-black cells than the left, or when the distribution is balanced)

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

    middle_row_count = sum(section_counts[3:6])
    left_count = sum(section_counts[0::3])
    right_count = sum(section_counts[2::3])
    total_count = sum(section_counts)

    if middle_row_count > total_count * 0.4:
        return create_output_grid("horizontal_stripe")
    elif right_count > left_count * 1.2 or abs(right_count - left_count) / total_count < 0.2:
        return create_output_grid("backwards_c")
    else:
        return create_output_grid("horizontal_stripe")
