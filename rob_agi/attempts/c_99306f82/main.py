from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_99306f82(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by following these steps:
    1. Identifies the blue (1) rectangle outline in the input grid.
    2. Collects the color sequence from the top-left corner diagonally.
    3. Fills the interior of the blue rectangle with concentric layers of colors
       from the collected sequence, starting from the outermost layer and moving inward.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the interior of the blue rectangle filled.
    """
    def find_blue_rectangle(grid: List[List[int]]) -> Tuple[int, int, int, int]:
        rows, cols = len(grid), len(grid[0])
        top, left = next((r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1)
        bottom = next(r for r in range(top + 1, rows) if grid[r][left] == 0) - 1
        right = next(c for c in range(left + 1, cols) if grid[top][c] == 0) - 1
        return top, left, bottom, right

    def collect_color_sequence(grid: List[List[int]], top: int, left: int) -> List[int]:
        sequence = []
        r, c = 0, 0
        while r < top and c < left:
            if grid[r][c] not in [0, 1]:
                sequence.append(grid[r][c])
            r += 1
            c += 1
        return sequence

    def fill_rectangle(grid: List[List[int]], top: int, left: int, bottom: int, right: int, sequence: List[int]) -> None:
        def fill_layer(r: int, c: int, color: int) -> None:
            while grid[r][c] == 0:
                grid[r][c] = color
                if r < bottom - 1 and grid[r + 1][c] == 0:
                    r += 1
                elif c < right - 1 and grid[r][c + 1] == 0:
                    c += 1
                elif r > top + 1 and grid[r - 1][c] == 0:
                    r -= 1
                elif c > left + 1 and grid[r][c - 1] == 0:
                    c -= 1
                else:
                    break

        r, c = top + 1, left + 1
        seq_index = 0
        while r <= bottom - 1 and c <= right - 1 and grid[r][c] == 0:
            fill_layer(r, c, sequence[seq_index % len(sequence)])
            r += 1
            c += 1
            seq_index += 1

    # Main logic
    output_grid = input_grid.deep_copy()
    grid = output_grid.values
    top, left, bottom, right = find_blue_rectangle(grid)
    color_sequence = collect_color_sequence(grid, top, left)
    fill_rectangle(grid, top, left, bottom, right, color_sequence)

    return output_grid
