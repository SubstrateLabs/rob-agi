from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_b457fec5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by filling gray areas with a diagonal pattern of colors.
    
    The solution follows these steps:
    1. Identify the color cluster and create an ordered list of colors.
    2. Analyze gray areas to determine the global fill direction.
    3. Sort gray cells from top to bottom, then left to right or right to left.
    4. Fill gray areas using a flood fill algorithm, maintaining the diagonal pattern.
    5. Ensure continuity across disconnected regions.
    6. Perform a final pass to fill any remaining gray cells.
    7. Return the transformed grid.
    
    The pattern starts from the top of each gray region, uses colors in the order they appear
    in the input, and follows a diagonal pattern while maintaining color sequence across regions.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    # Step 1: Identify color cluster
    color_sequence = [val for row in input_grid.values for val in row if val not in [0, 5]]
    if not color_sequence:
        return output_grid  # No colors to fill with
    
    # Step 2: Analyze gray areas and determine fill direction
    gray_cells = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 5]
    if not gray_cells:
        return output_grid  # No gray cells to fill
    
    avg_col = sum(c for _, c in gray_cells) / len(gray_cells)
    fill_direction = 1 if avg_col >= cols / 2 else -1
    
    # Step 3: Sort gray cells
    gray_cells.sort(key=lambda x: (x[0], -x[1] if fill_direction == -1 else x[1]))
    
    def get_next_color(index):
        return color_sequence[index % len(color_sequence)]
    
    def get_neighbors(r, c):
        return [(r+1, c), (r, c+fill_direction), (r+1, c+fill_direction)]
    
    # Step 4: Fill gray areas
    color_index = 0
    for start_r, start_c in gray_cells:
        if output_grid.values[start_r][start_c] == 5:
            queue = deque([(start_r, start_c)])
            while queue:
                r, c = queue.popleft()
                if 0 <= r < rows and 0 <= c < cols and output_grid.values[r][c] == 5:
                    output_grid.values[r][c] = get_next_color(color_index)
                    color_index += 1
                    neighbors = get_neighbors(r, c)
                    queue.extend(n for n in neighbors if 0 <= n[0] < rows and 0 <= n[1] < cols and output_grid.values[n[0]][n[1]] == 5)
    
    # Step 5 & 6: Final pass to fill any remaining gray cells
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] == 5:
                output_grid.values[r][c] = get_next_color(color_index)
                color_index += 1
    
    return output_grid
