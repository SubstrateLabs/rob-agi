from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_2c737e39(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by identifying the main pattern,
    creating a balanced duplicate, and removing isolated gray squares.

    1. Identify the main pattern using flood-fill from the top-left.
    2. Classify gray squares as connected or isolated.
    3. Calculate the balanced position for the duplicate.
    4. Create the duplicate pattern, omitting isolated gray squares.
    5. Remove all isolated gray squares.
    6. Adjust the duplicate if it's out of bounds.

    Args:
    input_grid (ColoredGrid): The input grid to transform.

    Returns:
    ColoredGrid: The transformed grid with the duplicated pattern.
    """
    def flood_fill(grid: List[List[int]], start: Tuple[int, int], visited: Set[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
        rows, cols = len(grid), len(grid[0])
        color = grid[start[0]][start[1]]
        if color == 0:
            return []
        
        pattern = []
        stack = [start]

        while stack:
            r, c = stack.pop()
            if (r, c) not in visited and 0 <= r < rows and 0 <= c < cols and grid[r][c] == color:
                visited.add((r, c))
                pattern.append((r, c, color))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    stack.append((r + dr, c + dc))
        
        return pattern

    def find_pattern(grid: List[List[int]]) -> List[Tuple[int, int, int]]:
        pattern = []
        visited = set()
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] != 0 and (r, c) not in visited:
                    pattern.extend(flood_fill(grid, (r, c), visited))
        return pattern

    def classify_gray_squares(pattern: List[Tuple[int, int, int]], grid: List[List[int]]) -> Set[Tuple[int, int]]:
        connected_grays = set((r, c) for r, c, color in pattern if color == 5)
        all_grays = set((r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == 5)
        return all_grays - connected_grays

    def calculate_center_of_mass(pattern: List[Tuple[int, int, int]]) -> Tuple[float, float]:
        if not pattern:
            return (0, 0)
        return sum(r for r, _, _ in pattern) / len(pattern), sum(c for _, c, _ in pattern) / len(pattern)

    def find_balanced_position(grid: List[List[int]], pattern: List[Tuple[int, int, int]]) -> Tuple[int, int]:
        rows, cols = len(grid), len(grid[0])
        com_r, com_c = calculate_center_of_mass(pattern)
        grid_center_r, grid_center_c = rows / 2, cols / 2
        shift_r = int(grid_center_r - com_r)
        shift_c = int(grid_center_c - com_c)
        return shift_r, shift_c

    input_values = input_grid.values
    pattern = find_pattern(input_values)
    isolated_grays = classify_gray_squares(pattern, input_values)
    shift_r, shift_c = find_balanced_position(input_values, pattern)

    output_grid = input_grid.deep_copy()
    output_values = output_grid.values

    # Create duplicate pattern
    for r, c, color in pattern:
        if (r, c) not in isolated_grays:
            new_r, new_c = r + shift_r, c + shift_c
            if 0 <= new_r < len(output_values) and 0 <= new_c < len(output_values[0]):
                output_values[new_r][new_c] = color

    # Remove all isolated gray squares
    for r, c in isolated_grays:
        output_values[r][c] = 0

    return output_grid
