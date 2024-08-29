from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque

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

def custom_sort(color, color_info):
    size, row, col = color_info[color]
    return (-size, -row, -col)

def experiment_sorting():
    # Test case from the first example
    input_grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 3, 0, 0, 0, 0, 0, 8, 0, 0],
        [0, 0, 3, 3, 3, 0, 0, 0, 8, 8, 0, 0],
        [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    rows, cols = len(input_grid), len(input_grid[0])
    color_info: Dict[int, Tuple[int, int, int]] = {}  # color: (size, row, col)

    for r in range(rows - 1, -1, -1):
        for c in range(cols - 1, -1, -1):
            color = input_grid[r][c]
            if color != 0:
                if color not in color_info:
                    size = find_largest_block(input_grid, color)
                    color_info[color] = (size, r, c)
                elif (r, c) == (color_info[color][1], color_info[color][2]):
                    size = find_largest_block(input_grid, color)
                    color_info[color] = (size, r, c)

    sorted_colors = sorted(color_info.keys(), key=lambda x: custom_sort(x, color_info))
    
    print("Color info:")
    for color, info in color_info.items():
        print(f"Color {color}: Size {info[0]}, Row {info[1]}, Col {info[2]}")
    
    print("\nSorted colors:")
    print(sorted_colors)

    result = [[color] for color in sorted_colors]
    print("\nFinal result:")
    print(result)

if __name__ == "__main__":
    experiment_sorting()
