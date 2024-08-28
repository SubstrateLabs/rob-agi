from rob_agi.colored_grid import ColoredGrid
from collections import defaultdict
from typing import List, Tuple, Dict

def solve_575b1a71(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing black squares with colored squares based on the following rules:
    1. Assigns colors (1-4) to black squares, prioritizing vertical alignment within columns.
    2. Avoids adjacent same-colored squares horizontally and vertically.
    3. Ensures all colors (1-4) are present in the output.
    4. Balances the number of squares for each color as much as possible.
    5. Handles isolated, edge, and corner squares specially.
    6. Enhances vertical alignment while maintaining color balance.
    7. Resolves conflicts and performs final adjustments to satisfy all conditions.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    total_black = sum(row.count(0) for row in output_grid.values)
    target_count = total_black // 4
    color_counter = {1: 0, 2: 0, 3: 0, 4: 0}

    # Identify special squares
    isolated_squares, edge_squares, corner_squares = identify_special_squares(output_grid)
    vertical_lines = identify_vertical_lines(output_grid)

    # Initial color assignment
    assign_colors_to_special_squares(output_grid, isolated_squares, edge_squares, corner_squares, color_counter)
    assign_colors_to_vertical_lines(output_grid, vertical_lines, color_counter)

    # Fill remaining squares
    fill_remaining_squares(output_grid, color_counter)

    # Ensure all colors are present
    ensure_all_colors_present(output_grid, color_counter)

    # Balance adjustment
    balance_colors(output_grid, color_counter)

    # Conflict resolution
    resolve_conflicts(output_grid, color_counter)

    # Enhance vertical alignment
    enhance_vertical_alignment(output_grid, color_counter)

    # Final check and adjustment
    final_check_and_adjust(output_grid, color_counter)

    return output_grid

def assign_vertical_colors(grid: ColoredGrid, color_counter: Dict[int, int], target_count: int):
    rows, cols = grid.get_dimensions()
    column_colors = {}
    for c in range(cols):
        black_squares = [(r, c) for r in range(rows) if grid.values[r][c] == 0]
        if len(black_squares) > 1:
            color = get_best_color(grid, black_squares, color_counter, target_count)
            column_colors[c] = color
            for r, _ in black_squares:
                if not has_conflict(grid, r, c, color):
                    grid.values[r][c] = color
                    color_counter[color] += 1

def get_best_color(grid: ColoredGrid, squares: List[Tuple[int, int]], color_counter: Dict[int, int], target_count: int) -> int:
    colors = [1, 2, 3, 4]
    return min(colors, key=lambda color: (
        max(color_counter.values()) - color_counter[color],
        sum(1 for r, c in squares if has_conflict(grid, r, c, color))
    ))

def fill_remaining_squares(grid: ColoredGrid, color_counter: Dict[int, int], target_count: int):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 0:
                color = get_best_color(grid, [(r, c)], color_counter, target_count)
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
            for r in range(grid.num_rows):
                for c in range(grid.num_cols):
                    if grid.values[r][c] != 5 and can_change_color(grid, r, c, color):
                        old_color = grid.values[r][c]
                        grid.values[r][c] = color
                        color_counter[color] += 1
                        color_counter[old_color] -= 1
                        return  # Exit after placing the missing color

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
def balance_colors(grid: ColoredGrid, color_counter: Dict[int, int], target_count: int):
    colors_to_reduce = [c for c in range(1, 5) if color_counter[c] > target_count]
    colors_to_increase = [c for c in range(1, 5) if color_counter[c] < target_count]
    
    for _ in range(min(len(colors_to_reduce), len(colors_to_increase))):
        color_to_reduce = colors_to_reduce.pop(0)
        color_to_increase = colors_to_increase.pop(0)
        
        for r in range(grid.num_rows):
            for c in range(grid.num_cols):
                if grid.values[r][c] == color_to_reduce and can_change_color(grid, r, c, color_to_increase):
                    grid.values[r][c] = color_to_increase
                    color_counter[color_to_reduce] -= 1
                    color_counter[color_to_increase] += 1
                    break
            if color_counter[color_to_reduce] == target_count:
                break

def handle_edge_corner_cases(grid: ColoredGrid, color_counter: Dict[int, int], target_count: int):
    rows, cols = grid.get_dimensions()
    edge_squares = ([(0, c) for c in range(cols)] + 
                    [(rows-1, c) for c in range(cols)] + 
                    [(r, 0) for r in range(1, rows-1)] + 
                    [(r, cols-1) for r in range(1, rows-1)])
    
    for r, c in edge_squares:
        if grid.values[r][c] != 5:
            best_color = get_best_color(grid, [(r, c)], color_counter, target_count)
            if best_color != grid.values[r][c]:
                color_counter[grid.values[r][c]] -= 1
                grid.values[r][c] = best_color
                color_counter[best_color] += 1

def fine_tune_solution(grid: ColoredGrid, color_counter: Dict[int, int], target_count: int):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != 5:
                best_color = get_best_color(grid, [(r, c)], color_counter, target_count)
                if best_color != grid.values[r][c] and can_change_color(grid, r, c, best_color):
                    color_counter[grid.values[r][c]] -= 1
                    grid.values[r][c] = best_color
                    color_counter[best_color] += 1

def resolve_conflicts(grid: ColoredGrid, color_counter: Dict[int, int]):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != 5:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid.values[r][c] == grid.values[nr][nc]:
                        new_color = get_least_used_color(color_counter, {grid.values[r][c]})
                        if can_change_color(grid, nr, nc, new_color):
                            color_counter[grid.values[nr][nc]] -= 1
                            grid.values[nr][nc] = new_color
                            color_counter[new_color] += 1
def identify_special_squares(grid: ColoredGrid) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    isolated = []
    edge = []
    corner = []
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 0:
                if is_isolated(grid, r, c):
                    isolated.append((r, c))
                elif r in (0, rows-1) or c in (0, cols-1):
                    if (r in (0, rows-1) and c in (0, cols-1)):
                        corner.append((r, c))
                    else:
                        edge.append((r, c))
    
    return isolated, edge, corner

def identify_vertical_lines(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    vertical_lines = []
    
    for c in range(cols):
        line = []
        for r in range(rows):
            if grid.values[r][c] == 0:
                line.append((r, c))
            elif line:
                if len(line) > 1:
                    vertical_lines.append(line)
                line = []
        if line and len(line) > 1:
            vertical_lines.append(line)
    
    return vertical_lines

def assign_colors_to_special_squares(grid: ColoredGrid, isolated: List[Tuple[int, int]], 
                                     edge: List[Tuple[int, int]], corner: List[Tuple[int, int]], 
                                     color_counter: Dict[int, int]):
    for r, c in isolated + corner + edge:
        if grid.values[r][c] == 0:
            color = get_best_color(grid, [(r, c)], color_counter)
            grid.values[r][c] = color
            color_counter[color] += 1

def assign_colors_to_vertical_lines(grid: ColoredGrid, vertical_lines: List[List[Tuple[int, int]]], 
                                    color_counter: Dict[int, int]):
    for line in vertical_lines:
        if len(line) > 2:
            colors = [1, 2] if sum(color_counter.values()) % 2 == 0 else [3, 4]
        else:
            colors = [get_best_color(grid, line, color_counter)]
        
        for i, (r, c) in enumerate(line):
            color = colors[i % len(colors)]
            if not has_conflict(grid, r, c, color):
                grid.values[r][c] = color
                color_counter[color] += 1

def enhance_vertical_alignment(grid: ColoredGrid, color_counter: Dict[int, int]):
    rows, cols = grid.get_dimensions()
    for c in range(cols):
        prev_color = None
        for r in range(rows):
            if grid.values[r][c] != 5:
                if prev_color and not has_conflict(grid, r, c, prev_color):
                    old_color = grid.values[r][c]
                    grid.values[r][c] = prev_color
                    color_counter[old_color] -= 1
                    color_counter[prev_color] += 1
                prev_color = grid.values[r][c]

def final_check_and_adjust(grid: ColoredGrid, color_counter: Dict[int, int]):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != 5:
                if has_conflict(grid, r, c, grid.values[r][c]):
                    new_color = get_best_color(grid, [(r, c)], color_counter)
                    color_counter[grid.values[r][c]] -= 1
                    grid.values[r][c] = new_color
                    color_counter[new_color] += 1

    missing_colors = [color for color in range(1, 5) if color_counter[color] == 0]
    for color in missing_colors:
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] != 5 and not has_conflict(grid, r, c, color):
                    old_color = grid.values[r][c]
                    grid.values[r][c] = color
                    color_counter[old_color] -= 1
                    color_counter[color] += 1
                    break
            if color_counter[color] > 0:
                break
