from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_b7cb93ac(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid into a 3x4 output grid based on color patterns.
    
    1. Analyzes the input grid to determine the top 2 most common colors (excluding sky blue).
    2. Creates a base 3x4 grid with the most common color.
    3. Places the second most common color either on the left (2x3 rectangle) or right (two 2x1 rectangles).
    4. Handles the sky blue color (8) if present, otherwise uses the third most common color.
    5. Ensures vertical symmetry in the output grid.
    
    Returns a new ColoredGrid object representing the transformed grid.
    """
    # Step 1: Analyze input grid
    color_counts = Counter(color for row in input_grid.values for color in row if color != 0 and color != 8)
    sky_blue_present = any(8 in row for row in input_grid.values)
    
    # Step 2: Determine color ranking
    top_colors = color_counts.most_common(2)
    while len(top_colors) < 2:
        top_colors.append((0, 0))  # Add black if fewer than 2 colors
    
    # Step 3: Create base output grid
    output = [[top_colors[0][0]] * 4 for _ in range(3)]
    
    # Step 4: Place second color
    second_color_pos = calculate_average_position(input_grid, top_colors[1][0])
    if is_left(second_color_pos, input_grid):
        place_left_rectangle(output, top_colors[1][0])
    else:
        place_right_rectangles(output, top_colors[1][0])
    
    # Step 5: Handle sky blue or third color
    third_color = 8 if sky_blue_present else (color_counts.most_common(3)[2][0] if len(color_counts) > 2 else 0)
    place_corners(output, third_color)
    
    # Step 6: Ensure vertical symmetry
    output[2] = output[0].copy()
    
    # Step 7: Create and return ColoredGrid
    return ColoredGrid(values=output)

def calculate_average_position(grid: ColoredGrid, color: int) -> Tuple[float, float]:
    positions = [(r, c) for r, row in enumerate(grid.values) for c, val in enumerate(row) if val == color]
    if not positions:
        return (0, 0)
    return (sum(r for r, _ in positions) / len(positions),
            sum(c for _, c in positions) / len(positions))

def is_left(pos: Tuple[float, float], grid: ColoredGrid) -> bool:
    _, cols = grid.get_dimensions()
    return pos[1] < cols / 2

def place_left_rectangle(grid: List[List[int]], color: int) -> None:
    for r in range(3):
        grid[r][0] = color
        grid[r][1] = color

def place_right_rectangles(grid: List[List[int]], color: int) -> None:
    grid[0][2] = color
    grid[0][3] = color
    grid[2][2] = color
    grid[2][3] = color

def place_corners(grid: List[List[int]], color: int) -> None:
    if grid[0][0] == grid[0][1]:  # 2x3 rectangle is on the right
        grid[0][0] = color
        grid[0][1] = color
        grid[2][0] = color
        grid[2][1] = color
    else:  # 2x1 rectangles are on the right
        grid[0][2] = color
        grid[0][3] = color
        grid[2][2] = color
        grid[2][3] = color
