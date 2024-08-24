from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque

def solve_f8ff0b80(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the f8ff0b80 challenge by identifying the largest contiguous blocks of each color
    and sorting them based on size, row, and column.

    The solution follows these steps:
    1. Find the largest contiguous block for each color using BFS.
    2. Traverse the grid from bottom-right to top-left, updating information for each color.
    3. Store the largest block size and bottom-rightmost position for each color.
    4. Sort the colors based on largest block size (descending), then row (descending), then column (descending).
    5. Format the sorted colors as a list of single-element lists.

    Args:
    input_grid (ColoredGrid): The input grid containing colored blocks.

    Returns:
    ColoredGrid: A new grid with sorted colors based on their largest block sizes and positions.
    """
    def find_largest_block(grid: List[List[int]], color: int) -> int:
        rows, cols = len(grid), len(grid[0])
        visited = set()
        max_size = 0
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color and (r, c) not in visited:
                    size = 0
                    queue = deque([(r, c)])
                    while queue:
                        curr_r, curr_c = queue.popleft()
                        if (curr_r, curr_c) in visited:
                            continue
                        visited.add((curr_r, curr_c))
                        size += 1
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            new_r, new_c = curr_r + dr, curr_c + dc
                            if 0 <= new_r < rows and 0 <= new_c < cols and grid[new_r][new_c] == color:
                                queue.append((new_r, new_c))
                    max_size = max(max_size, size)
        return max_size

    grid = input_grid.values
    rows, cols = len(grid), len(grid[0])
    color_info: Dict[int, Tuple[int, int, int]] = {}  # color: (size, row, col)

    for r in range(rows - 1, -1, -1):
        for c in range(cols - 1, -1, -1):
            color = grid[r][c]
            if color != 0:
                if color not in color_info:
                    size = find_largest_block(grid, color)
                    color_info[color] = (size, r, c)
                elif (r, c) == (color_info[color][1], color_info[color][2]):
                    size = find_largest_block(grid, color)
                    color_info[color] = (size, r, c)

    sorted_colors = sorted(color_info.keys(), key=lambda x: (-color_info[x][0], -color_info[x][1], -color_info[x][2]))
    result = [[color] for color in sorted_colors]
    return ColoredGrid(values=result)
