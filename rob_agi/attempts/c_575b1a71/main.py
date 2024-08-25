from rob_agi.colored_grid import ColoredGrid
from collections import defaultdict
from typing import List, Tuple, Dict

def solve_575b1a71(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing black squares with colored squares based on the following rules:
    1. Assigns colors (1-4) to black squares, prioritizing vertical alignment within columns.
    2. Avoids adjacent same-colored squares horizontally and vertically.
    3. Ensures all colors (1-4) are present in the output.
    4. Attempts to balance the number of squares for each color.
    5. Handles isolated squares and adjacent squares in rows.
    6. Fine-tunes the solution to improve vertical alignment and color balance.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    column_colors: Dict[int, int] = {}
    color_counter = {1: 0, 2: 0, 3: 0, 4: 0}

    assign_column_colors(output_grid, column_colors, color_counter)
    fill_isolated_squares(output_grid, color_counter)
    handle_adjacent_squares(output_grid, color_counter)
    ensure_all_colors_present(output_grid, color_counter)
    improve_vertical_alignment(output_grid, column_colors, color_counter)
    fine_tune_balance(output_grid, color_counter)

    return output_grid

def assign_column_colors(grid: ColoredGrid, column_colors: Dict[int, int], color_counter: Dict[int, int]):
    rows, cols = grid.get_dimensions()
    color_cycle = [1, 2, 3, 4]
    for c in range(cols):
        for r in range(rows):
            if grid.values[r][c] == 0:
                if c not in column_colors:
                    color = color_cycle[len(column_colors) % 4]
                    while has_conflict(grid, r, c, color):
                        color = color_cycle[(color_cycle.index(color) + 1) % 4]
                    column_colors[c] = color
                else:
                    color = column_colors[c]
                    if has_conflict(grid, r, c, color):
                        color = get_least_used_color(color_counter, {color})
                grid.values[r][c] = color
                color_counter[color] += 1

def fill_isolated_squares(grid: ColoredGrid, color_counter: Dict[int, int]):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 0 and is_isolated(grid, r, c):
                color = get_least_used_color(color_counter, get_conflicting_colors(grid, r, c))
                grid.values[r][c] = color
                color_counter[color] += 1

def handle_adjacent_squares(grid: ColoredGrid, color_counter: Dict[int, int]):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        c = 0
        while c < cols:
            if grid.values[r][c] == 0:
                start = c
                while c < cols and grid.values[r][c] == 0:
                    c += 1
                end = c
                assign_colors_to_row_segment(grid, r, start, end, color_counter)
            c += 1

def ensure_all_colors_present(grid: ColoredGrid, color_counter: Dict[int, int]):
    for color in range(1, 5):
        if color_counter[color] == 0:
            placed = False
            for r in range(grid.num_rows):
                for c in range(grid.num_cols):
                    if grid.values[r][c] != 5 and can_change_color(grid, r, c, color):
                        old_color = grid.values[r][c]
                        grid.values[r][c] = color
                        color_counter[color] += 1
                        color_counter[old_color] -= 1
                        placed = True
                        break
                if placed:
                    break

def improve_vertical_alignment(grid: ColoredGrid, column_colors: Dict[int, int], color_counter: Dict[int, int]):
    rows, cols = grid.get_dimensions()
    for c in range(cols):
        if c in column_colors:
            dominant_color = column_colors[c]
            for r in range(rows):
                if grid.values[r][c] != 5 and grid.values[r][c] != dominant_color:
                    if can_change_color(grid, r, c, dominant_color) and not significantly_worsens_balance(color_counter, grid.values[r][c], dominant_color):
                        color_counter[grid.values[r][c]] -= 1
                        color_counter[dominant_color] += 1
                        grid.values[r][c] = dominant_color

def fine_tune_balance(grid: ColoredGrid, color_counter: Dict[int, int]):
    target = sum(color_counter.values()) // 4
    colors_to_reduce = [color for color in range(1, 5) if color_counter[color] > target]
    colors_to_increase = [color for color in range(1, 5) if color_counter[color] < target]
    
    changes_made = 0
    max_changes = min(len(colors_to_reduce), len(colors_to_increase)) * 2
    
    while colors_to_reduce and colors_to_increase and changes_made < max_changes:
        color_to_reduce = colors_to_reduce[0]
        color_to_increase = colors_to_increase[0]
        
        changed = False
        for r in range(grid.num_rows):
            for c in range(grid.num_cols):
                if grid.values[r][c] == color_to_reduce and can_change_color(grid, r, c, color_to_increase):
                    grid.values[r][c] = color_to_increase
                    color_counter[color_to_reduce] -= 1
                    color_counter[color_to_increase] += 1
                    changes_made += 1
                    changed = True
                    break
            if changed:
                break
        
        if color_counter[color_to_reduce] == target:
            colors_to_reduce.pop(0)
        if color_counter[color_to_increase] == target:
            colors_to_increase.pop(0)

def has_conflict(grid: ColoredGrid, r: int, c: int, color: int) -> bool:
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols and grid.values[nr][nc] == color:
            return True
    return False

def get_least_used_color(color_counter: Dict[int, int], conflicts: set) -> int:
    return min((color for color in range(1, 5) if color not in conflicts), key=lambda x: color_counter[x])

def is_isolated(grid: ColoredGrid, r: int, c: int) -> bool:
    return all(grid.values[r+dr][c+dc] != 0 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)] if 0 <= r+dr < grid.num_rows and 0 <= c+dc < grid.num_cols)

def get_conflicting_colors(grid: ColoredGrid, r: int, c: int) -> set:
    return {grid.values[r+dr][c+dc] for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)] if 0 <= r+dr < grid.num_rows and 0 <= c+dc < grid.num_cols and grid.values[r+dr][c+dc] != 0 and grid.values[r+dr][c+dc] != 5}

def assign_colors_to_row_segment(grid: ColoredGrid, r: int, start: int, end: int, color_counter: Dict[int, int]):
    colors = [1, 2, 3, 4]
    for c in range(start, end):
        color = get_least_used_color(color_counter, get_conflicting_colors(grid, r, c))
        grid.values[r][c] = color
        color_counter[color] += 1
        colors.remove(color)
        if not colors:
            colors = [1, 2, 3, 4]

def can_change_color(grid: ColoredGrid, r: int, c: int, new_color: int) -> bool:
    return not has_conflict(grid, r, c, new_color)

def significantly_worsens_balance(color_counter: Dict[int, int], old_color: int, new_color: int) -> bool:
    old_count = color_counter[old_color]
    new_count = color_counter[new_color]
    return new_count - old_count > 2
