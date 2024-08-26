from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque

def solve_29700607(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by drawing lines connecting colored squares.
    
    For each color:
    1. Starts from the topmost occurrence of the color.
    2. Connects to other squares of the same color using vertical lines when possible.
    3. Makes horizontal turns when necessary to reach disconnected squares.
    4. Preserves intersections by not overwriting existing colors.
    5. Uses the minimum number of line segments to connect all squares of the same color.
    
    Returns a new ColoredGrid with the drawn lines.
    """
    output_grid = input_grid.deep_copy()
    color_positions = {}

    # Populate color_positions dictionary
    for row in range(len(input_grid.values)):
        for col in range(len(input_grid.values[0])):
            color = input_grid.values[row][col]
            if color != 0:  # Not black
                if color not in color_positions:
                    color_positions[color] = []
                color_positions[color].append((row, col))

    def find_next_coordinate(current: Tuple[int, int], remaining: List[Tuple[int, int]]) -> Tuple[int, int]:
        same_col = [coord for coord in remaining if coord[1] == current[1]]
        if same_col:
            return min(same_col, key=lambda x: abs(x[0] - current[0]))
        
        same_row = [coord for coord in remaining if coord[0] == current[0]]
        if same_row:
            return min(same_row, key=lambda x: abs(x[1] - current[1]))
        
        return min(remaining, key=lambda x: x[0])  # Topmost remaining

    for color, positions in color_positions.items():
        positions.sort()  # Sort by row, then column
        current = positions[0]
        queue = deque(positions[1:])

        while queue:
            next_coord = find_next_coordinate(current, list(queue))
            queue.remove(next_coord)

            # Draw vertical line
            start_row, end_row = min(current[0], next_coord[0]), max(current[0], next_coord[0])
            for row in range(start_row, end_row + 1):
                if output_grid.values[row][current[1]] == 0:
                    output_grid.values[row][current[1]] = color

            # Draw horizontal line if needed
            if current[1] != next_coord[1]:
                start_col, end_col = min(current[1], next_coord[1]), max(current[1], next_coord[1])
                for col in range(start_col, end_col + 1):
                    if output_grid.values[next_coord[0]][col] == 0:
                        output_grid.values[next_coord[0]][col] = color

            current = next_coord

    return output_grid
