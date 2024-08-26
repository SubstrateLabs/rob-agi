from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_b7cb93ac(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid into a 3x4 output grid based on color patterns.
    
    1. Analyzes the input grid to determine the top 3 most common colors.
    2. Creates a base 3x4 grid with the most common color.
    3. Places a 2x2 square of the second most common color in either top-right or bottom-left.
    4. Fills opposite corners with the third most common color.
    5. Adjusts the grid if necessary based on relative positions of colors in the input.
    
    Returns a new ColoredGrid object representing the transformed grid.
    """
    # Step 1: Analyze input grid
    color_counts = Counter(color for row in input_grid.values for color in row if color != 0)
    
    # Step 2: Determine color ranking
    top_colors = color_counts.most_common(3)
    while len(top_colors) < 3:
        top_colors.append((0, 0))  # Add black if fewer than 3 colors
    
    # Step 3: Create base output grid
    output = [[top_colors[0][0]] * 4 for _ in range(3)]
    
    # Step 4: Place second color
    second_color_pos = calculate_average_position(input_grid, top_colors[1][0])
    if is_top_right(second_color_pos, input_grid):
        place_2x2_square(output, top_colors[1][0], 0, 2)
    else:
        place_2x2_square(output, top_colors[1][0], 1, 0)
    
    # Step 5: Place third color
    place_corners(output, top_colors[2][0])
    
    # Step 6: Adjust if necessary
    if should_flip(input_grid, output, top_colors):
        output = flip_grid(output)
    
    # Step 7: Create and return ColoredGrid
    return ColoredGrid(values=output)

def calculate_average_position(grid: ColoredGrid, color: int) -> Tuple[float, float]:
    positions = [(r, c) for r, row in enumerate(grid.values) for c, val in enumerate(row) if val == color]
    if not positions:
        return (0, 0)
    return (sum(r for r, _ in positions) / len(positions),
            sum(c for _, c in positions) / len(positions))

def is_top_right(pos: Tuple[float, float], grid: ColoredGrid) -> bool:
    rows, cols = grid.get_dimensions()
    return pos[0] < rows / 2 and pos[1] >= cols / 2

def place_2x2_square(grid: List[List[int]], color: int, row: int, col: int) -> None:
    for r in range(row, row + 2):
        for c in range(col, col + 2):
            grid[r][c] = color

def place_corners(grid: List[List[int]], color: int) -> None:
    if grid[0][2] != grid[0][0]:  # 2x2 square is in top-right
        grid[2][0] = color
        grid[2][1] = color
    else:  # 2x2 square is in bottom-left
        grid[0][2] = color
        grid[0][3] = color

def should_flip(input_grid: ColoredGrid, output: List[List[int]], top_colors: List[Tuple[int, int]]) -> bool:
    input_positions = [calculate_average_position(input_grid, color) for color, _ in top_colors[:3]]
    output_positions = [calculate_average_position(ColoredGrid(values=output), color) for color, _ in top_colors[:3]]
    
    input_vertical_order = sorted(range(3), key=lambda i: input_positions[i][0])
    output_vertical_order = sorted(range(3), key=lambda i: output_positions[i][0])
    
    return input_vertical_order != output_vertical_order

def flip_grid(grid: List[List[int]]) -> List[List[int]]:
    return grid[::-1]
