from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def analyze_grid(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    print(f"Grid dimensions: {rows}x{cols}")

    # Count non-black cells and their colors
    color_count = {}
    non_black_count = 0
    for row in grid.values:
        for cell in row:
            if cell != 0:
                non_black_count += 1
                color_count[cell] = color_count.get(cell, 0) + 1

    print(f"Total non-black cells: {non_black_count}")
    print("Color distribution:")
    for color, count in sorted(color_count.items()):
        print(f"  Color {color}: {count} cells")

    # Analyze rows and columns for potential patterns
    def find_repeating_sequence(sequence: List[int]) -> Tuple[List[int], int]:
        for length in range(1, len(sequence) // 2 + 1):
            if sequence[:length] * (len(sequence) // length) == sequence[:len(sequence) - (len(sequence) % length)]:
                return sequence[:length], length
        return sequence, len(sequence)

    print("\nAnalyzing rows for patterns:")
    for i, row in enumerate(grid.values):
        pattern, length = find_repeating_sequence([cell for cell in row if cell != 0])
        print(f"Row {i}: pattern length = {length}, pattern = {pattern}")

    print("\nAnalyzing columns for patterns:")
    for j in range(cols):
        column = [grid.values[i][j] for i in range(rows)]
        pattern, length = find_repeating_sequence([cell for cell in column if cell != 0])
        print(f"Column {j}: pattern length = {length}, pattern = {pattern}")

# Example 1 input grid
example_1 = ColoredGrid(values=[
    [1, 3, 3, 1, 0, 0, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1],
    [3, 1, 5, 3, 0, 0, 3, 1, 5, 3, 1, 5, 3, 1, 5, 3, 1, 5, 3, 1, 5, 3],
    [3, 5, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 3],
    [1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1],
    [3, 1, 5, 3, 1, 5, 3, 1, 5, 3, 1, 5, 3, 1, 5, 0, 0, 0, 0, 0, 5, 3],
    [3, 5, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 0, 0, 0, 0, 0, 3, 3],
    [1, 3, 3, 1, 3, 3, 1, 3, 0, 0, 0, 0, 1, 3, 3, 0, 0, 0, 0, 0, 3, 1],
    [3, 1, 5, 3, 1, 5, 3, 1, 0, 0, 0, 0, 3, 1, 5, 0, 0, 0, 0, 0, 5, 3],
    [3, 5, 3, 3, 5, 3, 3, 5, 0, 0, 0, 0, 3, 5, 3, 3, 5, 3, 3, 5, 3, 3],
    [1, 3, 3, 1, 3, 3, 1, 3, 0, 0, 0, 0, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1],
    [3, 1, 5, 3, 1, 5, 3, 1, 0, 0, 0, 0, 3, 1, 5, 3, 1, 5, 3, 1, 5, 3],
    [3, 5, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 3],
    [1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1],
    [3, 1, 5, 3, 1, 5, 3, 1, 5, 3, 1, 5, 3, 1, 5, 3, 1, 5, 3, 1, 5, 3],
    [3, 5, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 3],
    [1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1],
    [3, 1, 5, 3, 1, 5, 3, 1, 5, 3, 1, 5, 3, 1, 5, 0, 0, 5, 3, 1, 5, 3],
    [0, 0, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 0, 0, 3, 3, 5, 3, 3],
    [0, 0, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1],
    [0, 0, 5, 3, 1, 5, 3, 1, 5, 3, 1, 5, 3, 1, 5, 3, 1, 5, 3, 1, 5, 3],
    [0, 0, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 3],
    [1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1]
])

analyze_grid(example_1)
