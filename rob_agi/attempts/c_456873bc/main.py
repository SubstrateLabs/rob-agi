from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_456873bc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Removes all green (3) areas, replacing them with black (0).
    2. Identifies red (2) lines and their intersections.
    3. Converts specific intersections of red lines to blue (8):
       - At grid edges
       - Where a red line terminates by meeting another red line perpendicularly
    4. Extends the pattern of red lines into previously green areas.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying all rules.
    """
    grid = input_grid.deep_copy()
    grid = remove_green_areas(grid)
    red_lines = identify_red_lines(grid)
    intersections = find_intersections(red_lines)
    classified_intersections = classify_intersections(intersections, grid)
    grid = convert_intersections_to_blue(grid, classified_intersections)
    empty_areas = find_empty_areas(grid)
    grid = extend_pattern(grid, empty_areas, red_lines)
    return grid

def remove_green_areas(grid: ColoredGrid) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 3:
                grid.set_cell(r, c, 0)
    return grid

def identify_red_lines(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    lines = []
    visited = set()

    def dfs(r: int, c: int) -> List[Tuple[int, int]]:
        line = []
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and grid.get_cell(curr_r, curr_c) == 2:
                visited.add((curr_r, curr_c))
                line.append((curr_r, curr_c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))
        return line

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.get_cell(r, c) == 2:
                line = dfs(r, c)
                if len(line) > 1:
                    lines.append(line)

    return lines

def find_intersections(lines: List[List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    intersections = set()
    for line in lines:
        for point in line:
            intersections.add(point)
    return list(intersections)

def classify_intersections(intersections: List[Tuple[int, int]], grid: ColoredGrid) -> List[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    classified = []
    for r, c in intersections:
        if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
            classified.append((r, c))
        else:
            neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)] 
                            if 0 <= r + dr < rows and 0 <= c + dc < cols and grid.get_cell(r + dr, c + dc) == 2)
            if neighbors == 1:
                classified.append((r, c))
    return classified

def convert_intersections_to_blue(grid: ColoredGrid, intersections: List[Tuple[int, int]]) -> ColoredGrid:
    for r, c in intersections:
        grid.set_cell(r, c, 8)
    return grid

def find_empty_areas(grid: ColoredGrid) -> List[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    return [(r, c) for r in range(rows) for c in range(cols) if grid.get_cell(r, c) == 0]

def extend_pattern(grid: ColoredGrid, empty_areas: List[Tuple[int, int]], red_lines: List[List[Tuple[int, int]]]) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    for r, c in empty_areas:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 2:
                grid.set_cell(r, c, 2)
                break
    
    # Re-run intersection classification and conversion for extended lines
    new_red_lines = identify_red_lines(grid)
    new_intersections = find_intersections(new_red_lines)
    new_classified_intersections = classify_intersections(new_intersections, grid)
    grid = convert_intersections_to_blue(grid, new_classified_intersections)
    
    return grid
