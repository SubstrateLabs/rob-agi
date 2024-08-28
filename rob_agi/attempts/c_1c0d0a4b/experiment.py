from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_1c0d0a4b.main import solve_1c0d0a4b

def analyze_grid(grid: ColoredGrid, color: int) -> dict:
    rows, cols = grid.get_dimensions()
    total_cells = rows * cols
    colored_cells = sum(row.count(color) for row in grid.values)
    return {
        "total_cells": total_cells,
        "colored_cells": colored_cells,
        "percentage": (colored_cells / total_cells) * 100
    }

def compare_grids(input_grid: ColoredGrid, output_grid: ColoredGrid):
    input_analysis = analyze_grid(input_grid, 8)
    output_analysis = analyze_grid(output_grid, 2)
    
    print("Input Grid Analysis (Sky Blue):")
    print(f"Total cells: {input_analysis['total_cells']}")
    print(f"Sky Blue cells: {input_analysis['colored_cells']}")
    print(f"Percentage: {input_analysis['percentage']:.2f}%")
    
    print("\nOutput Grid Analysis (Red):")
    print(f"Total cells: {output_analysis['total_cells']}")
    print(f"Red cells: {output_analysis['colored_cells']}")
    print(f"Percentage: {output_analysis['percentage']:.2f}%")
    
    print(f"\nReduction in marked cells: {input_analysis['colored_cells'] - output_analysis['colored_cells']}")
    print(f"Percentage reduction: {((input_analysis['colored_cells'] - output_analysis['colored_cells']) / input_analysis['colored_cells']) * 100:.2f}%")

# Example grids from the test cases
example_grids = [
    ColoredGrid(values=[
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 8, 0, 8, 0, 8, 0, 0, 0, 8, 8, 0, 0],
        [0, 8, 0, 8, 0, 0, 8, 0, 0, 0, 8, 0, 0],
        [0, 8, 8, 8, 0, 8, 8, 8, 0, 0, 8, 8, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]),
    ColoredGrid(values=[
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 0, 0],
        [0, 0, 8, 0, 0, 8, 8, 8, 0, 8, 8, 0, 0],
        [0, 8, 0, 8, 0, 8, 0, 8, 0, 0, 8, 8, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 0, 8, 8, 8, 0, 8, 8, 0, 0],
        [0, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0],
        [0, 0, 8, 0, 0, 8, 8, 8, 0, 8, 8, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]),
    ColoredGrid(values=[
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 0, 8, 0, 8, 0],
        [0, 0, 0, 0, 0, 0, 8, 8, 0],
        [0, 0, 8, 0, 0, 8, 0, 8, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 8, 8, 0, 0, 8, 0, 0, 0],
        [0, 0, 8, 8, 0, 8, 8, 8, 0],
        [0, 8, 0, 0, 0, 0, 8, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
]

for i, grid in enumerate(example_grids):
    print(f"\n--- Example {i} ---")
    output_grid = solve_1c0d0a4b(grid)
    compare_grids(grid, output_grid)
