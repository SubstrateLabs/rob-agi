from rob_agi.colored_grid import ColoredGrid

def is_non_black(cell: int) -> bool:
    return cell != 0

def has_horizontal_line(grid: ColoredGrid) -> bool:
    middle_row = grid.values[3]
    non_black_count = sum(1 for cell in middle_row if is_non_black(cell))
    return non_black_count >= 3  # Relaxed condition

def has_backwards_c_path(grid: ColoredGrid) -> bool:
    # Check top row (right to left)
    top_row = grid.values[0][::-1]
    top_non_black = sum(1 for cell in top_row if is_non_black(cell))
    
    # Check right column (top to bottom)
    right_col = [row[-1] for row in grid.values]
    right_non_black = sum(1 for cell in right_col if is_non_black(cell))
    
    # Check bottom row (left to right)
    bottom_row = grid.values[-1]
    bottom_non_black = sum(1 for cell in bottom_row if is_non_black(cell))
    
    # Check left column (top to bottom)
    left_col = [row[0] for row in grid.values]
    left_non_black = sum(1 for cell in left_col if is_non_black(cell))
    
    return (top_non_black >= 2 and right_non_black >= 2 and bottom_non_black >= 2 and left_non_black >= 2)

def create_output_grid(pattern: str) -> ColoredGrid:
    if pattern == "horizontal_stripe":
        return ColoredGrid(values=[[0,0,0],[8,8,8],[0,0,0]])
    else:  # backwards_c
        return ColoredGrid(values=[[0,8,8],[0,8,0],[0,8,0]])

def solve_9110e3c5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 7x7 input grid into a 3x3 output grid based on specific patterns.
    
    The function checks for two main patterns in the input grid:
    1. A horizontal line pattern, focusing on the middle row
    2. A backwards "C" path along the edges
    
    The backwards "C" pattern is prioritized over the horizontal line pattern.
    If neither pattern is clearly detected, it defaults to the horizontal stripe pattern.

    Args:
    input_grid (ColoredGrid): A 7x7 ColoredGrid object representing the input.

    Returns:
    ColoredGrid: A 3x3 ColoredGrid object representing the output pattern.
    """
    if has_backwards_c_path(input_grid):
        return create_output_grid("backwards_c")
    elif has_horizontal_line(input_grid):
        return create_output_grid("horizontal_stripe")
    else:
        return create_output_grid("horizontal_stripe")
