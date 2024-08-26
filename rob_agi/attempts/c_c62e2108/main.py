from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_c62e2108(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the c62e2108 challenge by replicating patterns within framed areas.
    
    The function identifies framed areas in the input grid, finds patterns within each area,
    replicates them horizontally and vertically to fill the framed area, preserves the
    original frame lines, and keeps unframed areas unchanged.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with replicated patterns within framed areas.
    """
    framed_areas = find_framed_areas(input_grid)
    new_grid = [row[:] for row in input_grid.values]
    
    for top, bottom, left, right in framed_areas:
        patterns = find_patterns(input_grid, top, bottom, left, right)
        if patterns:
            replicate_patterns(new_grid, patterns, top, bottom, left, right)
    
    preserve_frame_lines(new_grid, input_grid)
    return ColoredGrid(values=new_grid)

def find_framed_areas(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    framed_areas = []
    rows, cols = grid.get_dimensions()
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            if grid.values[i][j] == 1 and not visited[i][j]:  # Frame color is 1 (blue)
                top, bottom, left, right = i, i, j, j
                stack = [(i, j)]
                while stack:
                    r, c = stack.pop()
                    if 0 <= r < rows and 0 <= c < cols and grid.values[r][c] == 1 and not visited[r][c]:
                        visited[r][c] = True
                        top, bottom = min(top, r), max(bottom, r)
                        left, right = min(left, c), max(right, c)
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            stack.append((r + dr, c + dc))
                framed_areas.append((top, bottom, left, right))
    
    return framed_areas

def find_patterns(grid: ColoredGrid, top: int, bottom: int, left: int, right: int) -> List[Tuple[int, int, int, int, int]]:
    patterns = []
    for i in range(top + 1, bottom):
        for j in range(left + 1, right):
            if grid.values[i][j] != 0 and grid.values[i][j] != 1:
                color = grid.values[i][j]
                pattern_top, pattern_bottom, pattern_left, pattern_right = i, i, j, j
                while pattern_top > top + 1 and grid.values[pattern_top - 1][j] == color:
                    pattern_top -= 1
                while pattern_bottom < bottom - 1 and grid.values[pattern_bottom + 1][j] == color:
                    pattern_bottom += 1
                while pattern_left > left + 1 and grid.values[i][pattern_left - 1] == color:
                    pattern_left -= 1
                while pattern_right < right - 1 and grid.values[i][pattern_right + 1] == color:
                    pattern_right += 1
                patterns.append((pattern_top, pattern_bottom, pattern_left, pattern_right, color))
    return patterns

def replicate_patterns(new_grid: List[List[int]], patterns: List[Tuple[int, int, int, int, int]], top: int, bottom: int, left: int, right: int):
    for pattern_top, pattern_bottom, pattern_left, pattern_right, color in patterns:
        pattern_height = pattern_bottom - pattern_top + 1
        pattern_width = pattern_right - pattern_left + 1
        for row in range(top + 1, bottom, pattern_height):
            for col in range(left + 1, right, pattern_width):
                for i in range(pattern_height):
                    for j in range(pattern_width):
                        if top < row + i < bottom and left < col + j < right:
                            new_grid[row + i][col + j] = new_grid[pattern_top + i][pattern_left + j]

def preserve_frame_lines(new_grid: List[List[int]], original_grid: ColoredGrid):
    for i in range(len(original_grid.values)):
        for j in range(len(original_grid.values[0])):
            if original_grid.values[i][j] == 1:  # Preserve frame (blue color)
                new_grid[i][j] = original_grid.values[i][j]
