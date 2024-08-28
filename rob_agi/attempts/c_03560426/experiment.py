from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_03560426.main import extract_shapes, Shape
from typing import List

def analyze_shapes(grid: ColoredGrid) -> List[Shape]:
    shapes = extract_shapes(grid)
    for i, shape in enumerate(shapes):
        print(f"Shape {i + 1}:")
        print(f"  Color: {shape.color}")
        print(f"  Area: {shape.area}")
        print(f"  Width: {shape.width}")
        print(f"  Height: {shape.height}")
        print(f"  Coordinates: {shape.coords}")
        print()
    return shapes

# Example grids from the test cases
example_grids = [
    ColoredGrid(values=[
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 4],
        [1, 1, 0, 2, 2, 0, 3, 3, 0, 4],
        [1, 1, 0, 2, 2, 0, 3, 3, 0, 4]
    ]),
    ColoredGrid(values=[
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [8, 8, 8, 0, 0, 0, 0, 0, 0, 0],
        [8, 8, 8, 0, 0, 0, 0, 0, 0, 0],
        [8, 8, 8, 0, 7, 7, 0, 2, 2, 2],
        [8, 8, 8, 0, 7, 7, 0, 2, 2, 2]
    ]),
    ColoredGrid(values=[
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 3, 3, 3],
        [4, 4, 4, 4, 0, 2, 0, 3, 3, 3],
        [4, 4, 4, 4, 0, 2, 0, 3, 3, 3]
    ])
]

print("Analyzing shapes in example grids:")
for i, grid in enumerate(example_grids):
    print(f"\nExample {i + 1}:")
    analyze_shapes(grid)
