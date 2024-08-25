from rob_agi.colored_grid import ColoredGrid
from collections import defaultdict
from typing import List, Tuple

def solve_575b1a71(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing black squares with colored squares based on the following rules:
    1. Identifies connected regions of black squares.
    2. Assigns colors to regions based on their size and position.
    3. Ensures all colors (1-4) are present and balanced in the output.
    4. Maintains consistency by avoiding adjacent same-colored squares where possible.
    5. Preserves the pattern of vertical alignment for same-colored squares.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Find all black regions
    black_regions = output_grid.find_connected_regions(0)
    
    # Sort regions by column first, then by row
    black_regions.sort(key=lambda r: (min(c for _, c in r), min(r for r, _ in r)))
    
    # Color assignment
    color_count = defaultdict(int)
    color_sequence = [1, 2, 3, 4]  # Blue, Red, Green, Yellow
    color_index = 0
    
    for region in black_regions:
        color = color_sequence[color_index]
        for r, c in region:
            output_grid.values[r][c] = color
        color_count[color] += len(region)
        color_index = (color_index + 1) % 4
    
    # Ensure all colors are present and balance colors
    ensure_all_colors_present(output_grid, color_count)
    balance_colors(output_grid, color_count)
    
    return output_grid

def ensure_all_colors_present(grid: ColoredGrid, color_count: dict):
    for color in range(1, 5):
        if color_count[color] == 0:
            for r in range(grid.num_rows):
                for c in range(grid.num_cols):
                    if grid.values[r][c] != 5 and not has_same_color_neighbor(grid, r, c, color):
                        old_color = grid.values[r][c]
                        grid.values[r][c] = color
                        color_count[color] += 1
                        color_count[old_color] -= 1
                        return

def has_same_color_neighbor(grid: ColoredGrid, r: int, c: int, color: int) -> bool:
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols and grid.values[nr][nc] == color:
            return True
    return False

def balance_colors(grid: ColoredGrid, color_count: dict):
    target = sum(color_count.values()) // 4
    colors_to_reduce = [color for color in range(1, 5) if color_count[color] > target]
    colors_to_increase = [color for color in range(1, 5) if color_count[color] < target]
    
    while colors_to_reduce and colors_to_increase:
        color_to_reduce = colors_to_reduce[0]
        color_to_increase = colors_to_increase[0]
        
        for r in range(grid.num_rows):
            for c in range(grid.num_cols):
                if grid.values[r][c] == color_to_reduce and not has_same_color_neighbor(grid, r, c, color_to_increase):
                    grid.values[r][c] = color_to_increase
                    color_count[color_to_reduce] -= 1
                    color_count[color_to_increase] += 1
                    
                    if color_count[color_to_reduce] == target:
                        colors_to_reduce.pop(0)
                    if color_count[color_to_increase] == target:
                        colors_to_increase.pop(0)
                    
                    if not colors_to_reduce or not colors_to_increase:
                        return
                    break
            if not colors_to_reduce or not colors_to_increase:
                return
