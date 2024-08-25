from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_137f0df0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a red background, preserving gray blocks,
    and adding blue squares at specific intervals on the edges.
    
    1. Creates a full red (2) background.
    2. Transfers the original gray (5) blocks to their corresponding positions.
    3. Adds blue (1) squares on the edges based on the pattern of gray blocks:
       - Always on the bottom in gap columns.
       - On the left if the first gray group doesn't start at column 0.
       - On the right if the last gray group doesn't end at column 9.
       - On the top if there are gray squares in both the leftmost and rightmost columns.
    4. Fills remaining edge spaces with black (0).
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid according to the specified pattern.
    """
    def find_gray_column_groups(grid: List[List[int]]) -> List[Tuple[int, int]]:
        cols = len(grid[0])
        gray_cols = [any(row[c] == 5 for row in grid) for c in range(cols)]
        groups = []
        start = None
        for i, is_gray in enumerate(gray_cols):
            if is_gray and start is None:
                start = i
            elif not is_gray and start is not None:
                groups.append((start, i - 1))
                start = None
        if start is not None:
            groups.append((start, cols - 1))
        return groups

    def identify_gap_columns(gray_groups: List[Tuple[int, int]], cols: int) -> List[int]:
        gaps = []
        if gray_groups[0][0] > 0:
            gaps.extend(range(gray_groups[0][0]))
        for i in range(len(gray_groups) - 1):
            gaps.extend(range(gray_groups[i][1] + 1, gray_groups[i+1][0]))
        if gray_groups[-1][1] < cols - 1:
            gaps.extend(range(gray_groups[-1][1] + 1, cols))
        return gaps

    def create_transformed_grid(input_grid: List[List[int]], gray_groups: List[Tuple[int, int]]) -> List[List[int]]:
        rows, cols = len(input_grid), len(input_grid[0])
        new_grid = [[2 for _ in range(cols)] for _ in range(rows)]  # Fill with red

        # Transfer gray blocks
        for r in range(rows):
            for c in range(cols):
                if input_grid[r][c] == 5:
                    new_grid[r][c] = 5

        gap_columns = identify_gap_columns(gray_groups, cols)

        # Place blue squares
        for c in gap_columns:
            new_grid[rows-1][c] = 1  # Bottom edge
        if gray_groups[0][0] > 0:
            for r in range(rows):
                new_grid[r][0] = 1  # Left edge
        if gray_groups[-1][1] < cols - 1:
            for r in range(rows):
                new_grid[r][cols-1] = 1  # Right edge
        if gray_groups[0][0] == 0 and gray_groups[-1][1] == cols - 1:
            for c in gap_columns:
                new_grid[0][c] = 1  # Top edge

        # Fill remaining edge spaces with black
        for c in range(cols):
            if new_grid[0][c] == 2:
                new_grid[0][c] = 0
            if new_grid[rows-1][c] == 2:
                new_grid[rows-1][c] = 0
        for r in range(rows):
            if new_grid[r][0] == 2:
                new_grid[r][0] = 0
            if new_grid[r][cols-1] == 2:
                new_grid[r][cols-1] = 0

        return new_grid

    input_values = input_grid.values
    gray_groups = find_gray_column_groups(input_values)
    new_grid = create_transformed_grid(input_values, gray_groups)
    
    return ColoredGrid(values=new_grid)
