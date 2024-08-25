from rob_agi.colored_grid import ColoredGrid
from collections import deque

def solve_58e15b12(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored squares diagonally.
    
    The algorithm works as follows:
    1. Identifies all non-black squares in the input grid.
    2. Expands these squares diagonally, alternating colors (8 and 3).
    3. Creates intersections (color 6) where expansions meet.
    4. Preserves original colored squares.
    5. Uses a distance-based cutoff to limit expansion.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    original_squares = []
    expansion_queue = deque()
    
    # Initialize expansion queue and original squares
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                original_squares.append((r, c, input_grid.values[r][c]))
                expansion_queue.append((r, c, input_grid.values[r][c], 0))
    
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    max_distance = max(rows, cols) // 2  # Distance-based cutoff
    
    # Expansion algorithm
    while expansion_queue:
        r, c, color, distance = expansion_queue.popleft()
        if distance >= max_distance:
            continue
        
        for dr, dc in directions:
            new_r, new_c = r + dr, c + dc
            if 0 <= new_r < rows and 0 <= new_c < cols:
                current_color = output_grid.values[new_r][new_c]
                if current_color == 0:
                    new_color = 3 if color == 8 else 8
                    output_grid.values[new_r][new_c] = new_color
                    expansion_queue.append((new_r, new_c, new_color, distance + 1))
                elif current_color != color and current_color != 6:
                    output_grid.values[new_r][new_c] = 6
    
    # Restore original squares
    for r, c, color in original_squares:
        output_grid.values[r][c] = color
    
    return output_grid
