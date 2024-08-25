from rob_agi.colored_grid import ColoredGrid
from collections import defaultdict
from typing import List, Tuple

def solve_575b1a71(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing black squares with colored squares based on the following rules:
    1. Assigns colors (1-4) to black squares in a left-to-right, top-to-bottom order.
    2. Maintains vertical alignment of colors within columns where possible.
    3. Avoids adjacent same-colored squares horizontally and vertically.
    4. Ensures all colors (1-4) are present in the output.
    5. Attempts to balance the number of squares for each color.
    6. Prioritizes vertical alignment over perfect color balance.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    color_cycle = [1, 2, 3, 4]  # Blue, Red, Green, Yellow
    column_colors = {}
    color_count = defaultdict(int)

    # First pass - Assign colors
    for c in range(cols):
        for r in range(rows):
            if output_grid.values[r][c] == 0:
                if c not in column_colors:
                    color = color_cycle[len(column_colors) % 4]
                    column_colors[c] = color
                else:
                    color = column_colors[c]
                
                # Check for conflicts and resolve
                while has_same_color_neighbor(output_grid, r, c, color):
                    color = color_cycle[(color_cycle.index(color) + 1) % 4]
                
                output_grid.values[r][c] = color
                color_count[color] += 1
                column_colors[c] = color

    # Second pass - Resolve conflicts and ensure all colors are present
    ensure_all_colors_present(output_grid, color_count)

    # Third pass - Balance colors and maintain vertical alignment
    balance_colors(output_grid, color_count)
    align_vertically(output_grid, column_colors)

    return output_grid

def align_vertically(grid: ColoredGrid, column_colors: Dict[int, int]):
    rows, cols = grid.get_dimensions()
    for c in range(cols):
        if c in column_colors:
            dominant_color = column_colors[c]
            for r in range(rows):
                if grid.values[r][c] != 5 and grid.values[r][c] != dominant_color:
                    if not has_same_color_neighbor(grid, r, c, dominant_color):
                        grid.values[r][c] = dominant_color

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
                    if not (r > 0 and grid.values[r-1][c] != 5 and grid.values[r-1][c] != color_to_reduce):
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
