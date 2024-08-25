from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e21a174a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the e21a174a challenge by rearranging color groups vertically.
    
    The function identifies distinct color groups in the input grid,
    preserves their internal structure, and then rearranges them from bottom to top
    based on their lowest point. Non-connected cells of the same color are treated
    as separate groups. Empty space (black) is maintained at the top of the grid.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with color groups rearranged.
    """
    rows, cols = input_grid.get_dimensions()
    
    def get_connected_group(start_row: int, start_col: int, color: int) -> List[Tuple[int, int]]:
        group = []
        stack = [(start_row, start_col)]
        visited = set()
        
        while stack:
            r, c = stack.pop()
            if (r, c) not in visited and 0 <= r < rows and 0 <= c < cols and input_grid.values[r][c] == color:
                visited.add((r, c))
                group.append((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    stack.append((r + dr, c + dc))
        
        return group

    # Step 1: Identify connected color groups
    color_groups = []
    visited = set()

    for row in range(rows):
        for col in range(cols):
            if (row, col) not in visited and input_grid.values[row][col] != 0:
                group = get_connected_group(row, col, input_grid.values[row][col])
                color_groups.append({
                    'color': input_grid.values[row][col],
                    'cells': group,
                    'bottom': max(r for r, _ in group)
                })
                visited.update(group)

    # Step 2: Sort color groups from bottom to top
    color_groups.sort(key=lambda g: g['bottom'], reverse=True)

    # Step 3: Create the output grid
    output_values = [[0 for _ in range(cols)] for _ in range(rows)]
    current_row = rows - 1

    # Step 4: Place color groups in new positions
    for group in color_groups:
        group_height = max(r for r, _ in group['cells']) - min(r for r, _ in group['cells']) + 1
        new_bottom = current_row
        shift = new_bottom - group['bottom']

        for old_row, col in group['cells']:
            new_row = old_row + shift
            output_values[new_row][col] = group['color']

        current_row = new_bottom - group_height

    # Step 5: Return the new ColoredGrid
    return ColoredGrid(values=output_values)
