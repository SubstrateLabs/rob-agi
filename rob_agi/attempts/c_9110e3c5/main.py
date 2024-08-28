from rob_agi.colored_grid import ColoredGrid

def is_non_black(cell: int) -> bool:
    return cell != 0

def edge_strength(edge: list) -> float:
    return sum(1 for cell in edge if is_non_black(cell)) / len(edge)

def has_vertical_line(grid: ColoredGrid) -> bool:
    second_to_last_col = [row[-2] for row in grid.values]
    return edge_strength(second_to_last_col) >= 4/7

def determine_pattern(grid: ColoredGrid) -> str:
    top_edge = grid.values[0]
    right_edge = [row[-1] for row in grid.values]
    bottom_edge = grid.values[-1]
    left_edge = [row[0] for row in grid.values]
    
    edges = [top_edge, right_edge, bottom_edge, left_edge]
    strong_edges = sum(1 for edge in edges if edge_strength(edge) >= 4/7)
    
    if strong_edges >= 3:
        if edge_strength(right_edge) >= 4/7 or has_vertical_line(grid):
            return "backwards_c"
        else:
            return "inverted_l"
    else:
        return "horizontal_stripe"

def create_output_grid(pattern: str) -> ColoredGrid:
    if pattern == "horizontal_stripe":
        return ColoredGrid(values=[[0,0,0],[8,8,8],[0,0,0]])
    elif pattern == "backwards_c":
        return ColoredGrid(values=[[0,8,8],[0,8,0],[0,8,0]])
    else:  # inverted_l
        return ColoredGrid(values=[[8,8,8],[8,0,0],[8,0,0]])

def solve_9110e3c5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 7x7 input grid into a 3x3 output grid based on specific patterns.
    
    The function analyzes the edges and second-to-last column of the input grid to determine the output pattern:
    1. If at least 3 edges have a strong presence of non-black cells (>=4/7):
       a. If the right edge is strong or there's a vertical line in the second-to-last column:
          Output a backwards "C" pattern.
       b. Otherwise, output an inverted "L" pattern.
    2. If fewer than 3 edges have a strong presence:
       Output a horizontal stripe pattern.

    Args:
    input_grid (ColoredGrid): A 7x7 ColoredGrid object representing the input.

    Returns:
    ColoredGrid: A 3x3 ColoredGrid object representing the output pattern.
    """
    pattern = determine_pattern(input_grid)
    return create_output_grid(pattern)
