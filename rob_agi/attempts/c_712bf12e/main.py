from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_712bf12e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating red paths from bottom red squares.
    
    The function identifies red squares in the bottom row and creates upward paths for each,
    avoiding gray squares and other red paths. Paths grow simultaneously, prioritizing
    leftmost paths in case of conflicts. Paths can slightly deviate within their lanes
    to navigate around obstacles.
    
    Args:
    input_grid (ColoredGrid): The input grid with initial red and gray squares.
    
    Returns:
    ColoredGrid: The transformed grid with red paths added.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Find starting points (red squares in the bottom row)
    start_points = [(rows-1, c) for c in range(cols) if output_grid.get_cell(rows-1, c) == 2]
    
    # Define paths and their current positions
    paths = [[(r, c)] for r, c in start_points]
    
    # Define lane boundaries (2 columns on each side)
    lane_width = 5
    
    def is_valid_move(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and output_grid.get_cell(r, c) not in [2, 5]
    
    # Grow paths simultaneously
    for current_row in range(rows-2, -1, -1):
        for i, path in enumerate(paths):
            if not path or path[-1][0] == 0:  # Skip if path is complete or empty
                continue
            
            r, c = path[-1]
            lane_left = max(0, c - lane_width // 2)
            lane_right = min(cols - 1, c + lane_width // 2)
            
            # Try moving up
            if is_valid_move(current_row, c):
                path.append((current_row, c))
            else:
                # Try moving left or right within the lane
                for dc in range(1, lane_width):
                    if lane_left <= c - dc and is_valid_move(current_row, c - dc):
                        path.append((current_row, c - dc))
                        break
                    if c + dc <= lane_right and is_valid_move(current_row, c + dc):
                        path.append((current_row, c + dc))
                        break
            
            # Mark the new position as red
            if len(path) > len(paths[i]):
                output_grid.set_cell(path[-1][0], path[-1][1], 2)
    
    return output_grid
