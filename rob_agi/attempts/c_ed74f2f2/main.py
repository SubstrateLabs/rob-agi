from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ed74f2f2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 9x5 input grid into a 3x3 output grid based on the following rules:
    1. Divides the input grid into nine 3x3 sections (with some overlap for the bottom row).
    2. Analyzes each section for the presence of gray (5) cells.
    3. Creates a 3x3 boolean grid marking sections with sufficient gray cells.
    4. Analyzes the pattern in the boolean grid.
    5. Determines the final color based on the recognized pattern:
       - Simple shapes (C, L, straight line, including rotations) use red (2)
       - Complex connected shapes use blue (1)
       - Disconnected or very complex shapes use green (3)
    6. Creates the final 3x3 ColoredGrid output with the determined color.
    """
    boolean_grid = analyze_grid(input_grid)
    pattern = determine_pattern(boolean_grid)
    color = determine_color(pattern)
    return create_final_output(boolean_grid, color)

def analyze_grid(grid: ColoredGrid) -> List[List[bool]]:
    boolean_grid = []
    for i in range(3):
        row = []
        for j in range(3):
            section = grid.extract_subgrid(i*2, j*3, 3, 3)
            row.append(analyze_section(section))
        boolean_grid.append(row)
    return boolean_grid

def analyze_section(section: ColoredGrid) -> bool:
    return sum(cell == 5 for row in section.values for cell in row) >= 2

def determine_pattern(grid: List[List[bool]]) -> str:
    if is_simple_shape(grid):
        return "simple"
    elif is_complex_connected(grid):
        return "complex_connected"
    else:
        return "disconnected_or_very_complex"

def is_simple_shape(grid: List[List[bool]]) -> bool:
    patterns = [
        [[True,True,True],[True,False,False],[True,True,True]],  # C
        [[True,True,True],[True,False,False],[True,False,False]],  # L
        [[True,True,True],[False,False,False],[False,False,False]],  # Horizontal line
        [[True,False,False],[True,False,False],[True,False,False]],  # Vertical line
        [[True,False,False],[False,True,False],[False,False,True]],  # Diagonal
    ]
    return any(grid == pattern or grid == [row[::-1] for row in pattern] for pattern in patterns)

def is_complex_connected(grid: List[List[bool]]) -> bool:
    visited = set()
    def dfs(i, j):
        if not (0 <= i < 3 and 0 <= j < 3) or not grid[i][j] or (i, j) in visited:
            return
        visited.add((i, j))
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            dfs(i + di, j + dj)
    
    start = next((i, j) for i, row in enumerate(grid) for j, cell in enumerate(row) if cell)
    dfs(*start)
    return len(visited) > 3 and len(visited) == sum(sum(row) for row in grid)

def determine_color(pattern: str) -> int:
    if pattern == "simple":
        return 2  # Red
    elif pattern == "complex_connected":
        return 1  # Blue
    else:
        return 3  # Green

def create_final_output(boolean_grid: List[List[bool]], color: int) -> ColoredGrid:
    return ColoredGrid(values=[[color if cell else 0 for cell in row] for row in boolean_grid])
