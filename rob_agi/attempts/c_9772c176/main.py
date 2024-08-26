from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List
import random

def solve_9772c176(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding a yellow (4) path that responds to sky blue (8) shapes.
    
    The solution follows these steps:
    1. Create a continuous yellow path from top-left to bottom-right.
    2. The path follows the edges of sky blue shapes when encountered.
    3. Add "rays" and additional yellow pixels to create visual interest.
    4. Handle empty or near-empty grids with a meandering path.
    5. Ensure all yellow pixels are connected in the final result.

    Args:
    input_grid (ColoredGrid): The input grid containing sky blue shapes.

    Returns:
    ColoredGrid: The transformed grid with added yellow paths.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    def is_valid_move(x: int, y: int) -> bool:
        return 0 <= x < rows and 0 <= y < cols and output_grid.get_cell(x, y) != 8

    def get_neighbors(x: int, y: int) -> List[Tuple[int, int]]:
        return [(x+dx, y+dy) for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
                if is_valid_move(x+dx, y+dy)]

    def is_adjacent_to_blue(x: int, y: int) -> bool:
        return any(0 <= x+dx < rows and 0 <= y+dy < cols and output_grid.get_cell(x+dx, y+dy) == 8
                   for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)])

    def choose_move(x: int, y: int, target_x: int, target_y: int) -> Tuple[int, int]:
        neighbors = get_neighbors(x, y)
        if not neighbors:
            return x, y  # Stay in place if no valid moves
        
        # Prioritize moves along blue shapes
        blue_adjacent = [n for n in neighbors if is_adjacent_to_blue(*n)]
        if blue_adjacent:
            return random.choice(blue_adjacent)
        
        # Otherwise, move towards the target
        dx = target_x - x
        dy = target_y - y
        preferred = [n for n in neighbors if (n[0]-x)*dx + (n[1]-y)*dy > 0]
        return random.choice(preferred) if preferred else random.choice(neighbors)

    def create_path():
        x, y = 0, 0
        output_grid.set_cell(x, y, 4)
        while (x, y) != (rows-1, cols-1):
            nx, ny = choose_move(x, y, rows-1, cols-1)
            if (nx, ny) != (x, y):
                output_grid.set_cell(nx, ny, 4)
            x, y = nx, ny

    def add_rays():
        for x in range(rows):
            for y in range(cols):
                if output_grid.get_cell(x, y) == 4:
                    for _ in range(2):  # Add up to 2 rays per yellow pixel
                        nx, ny = random.choice(get_neighbors(x, y))
                        if output_grid.get_cell(nx, ny) == 0:
                            output_grid.set_cell(nx, ny, 4)

    def ensure_connectivity():
        def dfs(x, y):
            stack = [(x, y)]
            visited = set()
            while stack:
                cx, cy = stack.pop()
                if (cx, cy) in visited:
                    continue
                visited.add((cx, cy))
                for nx, ny in get_neighbors(cx, cy):
                    if output_grid.get_cell(nx, ny) == 4:
                        stack.append((nx, ny))
            return visited

        connected = dfs(0, 0)
        for x in range(rows):
            for y in range(cols):
                if output_grid.get_cell(x, y) == 4 and (x, y) not in connected:
                    output_grid.set_cell(x, y, 0)  # Remove disconnected yellow pixels

    # Main execution
    create_path()
    add_rays()
    ensure_connectivity()

    return output_grid
