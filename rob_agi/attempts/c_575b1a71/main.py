from rob_agi.colored_grid import ColoredGrid
from collections import defaultdict, deque
from typing import List, Tuple

def solve_575b1a71(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing black squares with colored squares based on the following rules:
    1. Identifies connected regions of black squares.
    2. Assigns colors to regions based on their size and shape.
    3. Ensures all colors (1-4) are present and balanced in the output.
    4. Maintains consistency by avoiding adjacent same-colored squares (except for large yellow regions).
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Find all black regions
    black_regions = output_grid.find_connected_regions(0)
    
    # Sort regions by size (descending) and then by top-left position
    black_regions.sort(key=lambda r: (-len(r), min(r)))
    
    # Color assignment
    color_count = defaultdict(int)
    for region in black_regions:
        size = len(region)
        if size >= 4:
            color = 4  # Yellow for large regions
        elif size == 3:
            if is_line(region):
                color = 4  # Yellow for line of 3
            else:
                for i, (r, c) in enumerate(region):
                    output_grid.values[r][c] = 2 if i < 2 else 3
                color_count[2] += 2
                color_count[3] += 1
                continue
        elif size == 2:
            output_grid.values[region[0][0]][region[0][1]] = 1
            output_grid.values[region[1][0]][region[1][1]] = 2
            color_count[1] += 1
            color_count[2] += 1
            continue
        else:
            color = next_color(color_count)
        
        for r, c in region:
            output_grid.values[r][c] = color
        color_count[color] += size
    
    # Ensure all colors are present
    for color in range(1, 5):
        if color_count[color] == 0:
            add_missing_color(output_grid, color, color_count)
    
    # Balance color distribution
    balance_colors(output_grid, color_count)
    
    return output_grid

def is_line(region: List[Tuple[int, int]]) -> bool:
    return len(set(r for r, _ in region)) == 1 or len(set(c for _, c in region)) == 1

def next_color(color_count: dict) -> int:
    return min(range(1, 4), key=lambda c: color_count[c])

def add_missing_color(grid: ColoredGrid, color: int, color_count: dict):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 5 and has_colored_neighbor(grid, r, c):
                grid.values[r][c] = color
                color_count[color] += 1
                return
    
    # If no suitable gray square, replace a blue or red square
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] in [1, 2]:
                old_color = grid.values[r][c]
                grid.values[r][c] = color
                color_count[color] += 1
                color_count[old_color] -= 1
                return

def has_colored_neighbor(grid: ColoredGrid, r: int, c: int) -> bool:
    rows, cols = grid.get_dimensions()
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] not in [0, 5]:
            return True
    return False

def balance_colors(grid: ColoredGrid, color_count: dict):
    rows, cols = grid.get_dimensions()
    target = sum(color_count.values()) // 4
    
    for color in range(1, 5):
        while color_count[color] > target + 1:
            for r in range(rows):
                for c in range(cols):
                    if grid.values[r][c] == color:
                        new_color = min(range(1, 5), key=lambda c: color_count[c])
                        if new_color != color and is_valid_change(grid, r, c, new_color):
                            grid.values[r][c] = new_color
                            color_count[color] -= 1
                            color_count[new_color] += 1
                            if color_count[color] <= target + 1:
                                break
                if color_count[color] <= target + 1:
                    break

def is_valid_change(grid: ColoredGrid, r: int, c: int, new_color: int) -> bool:
    rows, cols = grid.get_dimensions()
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == new_color:
            return False
    return True
