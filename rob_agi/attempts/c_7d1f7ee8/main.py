from rob_agi.colored_grid import ColoredGrid
from typing import List, Set, Tuple

def solve_7d1f7ee8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling the interiors of outermost frames with their color.
    
    The function works as follows:
    1. Identifies outermost frames by finding non-black cells on the grid border.
    2. For each outermost frame, fills its interior with its color.
    3. Preserves the black background and areas outside of any frames.
    
    Args:
        input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
        ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == rows - 1 or c == 0 or c == cols - 1
    
    def flood_fill_frame(r: int, c: int, color: int):
        stack = [(r, c)]
        frame = set()
        while stack:
            x, y = stack.pop()
            if (x, y) in frame or output_grid.values[x][y] != color:
                continue
            frame.add((x, y))
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    stack.append((nx, ny))
        return frame
    
    def flood_fill_interior(frame: Set[Tuple[int, int]], color: int):
        for r, c in frame:
            stack = [(r, c)]
            while stack:
                x, y = stack.pop()
                if output_grid.values[x][y] == 0:
                    output_grid.values[x][y] = color
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in frame:
                            stack.append((nx, ny))
    
    # Identify and process outermost frames
    for r in range(rows):
        for c in range(cols):
            if is_border(r, c) and output_grid.values[r][c] != 0:
                color = output_grid.values[r][c]
                frame = flood_fill_frame(r, c, color)
                flood_fill_interior(frame, color)
    
    return output_grid
