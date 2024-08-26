from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_3ee1011a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid into an output grid based on the following rules:
    1. Identifies the longest continuous line of a single color in the input.
    2. Creates an output grid with dimensions equal to this longest line (minimum 5x5).
    3. Fills the output grid with concentric squares of colors found in the input.
    4. The outermost color is that of the longest line.
    5. Inner colors are ordered based on the size of their largest continuous group.
    6. The innermost color is from the smallest group in the input.
    7. If the innermost color formed a 2-pixel group in the input, it becomes a 2x2 square in the output center.
    """
    
    def find_longest_line(grid: ColoredGrid) -> Tuple[int, int]:
        longest = 0
        color = 0
        rows, cols = grid.get_dimensions()
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] != 0:
                    # Check horizontal
                    length = 1
                    while c + length < cols and grid.values[r][c + length] == grid.values[r][c]:
                        length += 1
                    if length > longest:
                        longest = length
                        color = grid.values[r][c]
                    # Check vertical
                    length = 1
                    while r + length < rows and grid.values[r + length][c] == grid.values[r][c]:
                        length += 1
                    if length > longest:
                        longest = length
                        color = grid.values[r][c]
        return max(longest, 5), color  # Ensure minimum size of 5

    def analyze_input(grid: ColoredGrid) -> Dict[int, Dict]:
        color_info = {}
        rows, cols = grid.get_dimensions()
        for r in range(rows):
            for c in range(cols):
                color = grid.values[r][c]
                if color != 0:
                    if color not in color_info:
                        color_info[color] = {"first": (r, c), "max_group": 1, "has_2_group": False}
                    # Check horizontal group
                    length = 1
                    while c + length < cols and grid.values[r][c + length] == color:
                        length += 1
                    color_info[color]["max_group"] = max(color_info[color]["max_group"], length)
                    if length == 2:
                        color_info[color]["has_2_group"] = True
                    # Check vertical group
                    length = 1
                    while r + length < rows and grid.values[r + length][c] == color:
                        length += 1
                    color_info[color]["max_group"] = max(color_info[color]["max_group"], length)
                    if length == 2:
                        color_info[color]["has_2_group"] = True
        return color_info

    def create_output_grid(size: int, colors: List[int]) -> ColoredGrid:
        grid = ColoredGrid(values=[[colors[0]] * size for _ in range(size)])
        for i, color in enumerate(colors[1:], 1):
            square_size = size - 2 * i
            if square_size <= 0:
                break
            for r in range(i, i + square_size):
                for c in range(i, i + square_size):
                    grid.values[r][c] = color
        return grid

    # Find the longest line and its color
    size, outer_color = find_longest_line(input_grid)
    
    # Analyze the input grid
    color_info = analyze_input(input_grid)
    
    # Determine the color order
    color_order = [outer_color]
    for color in sorted(color_info, key=lambda x: (-color_info[x]["max_group"], color_info[x]["first"])):
        if color != outer_color:
            color_order.append(color)
    
    # Create the output grid
    output_grid = create_output_grid(size, color_order)
    
    # Handle 2x2 center if needed
    innermost_color = color_order[-1]
    if color_info[innermost_color]["has_2_group"] and size >= 4:
        center = size // 2 - 1
        output_grid.values[center][center] = innermost_color
        output_grid.values[center][center+1] = innermost_color
        output_grid.values[center+1][center] = innermost_color
        output_grid.values[center+1][center+1] = innermost_color
    
    return output_grid
