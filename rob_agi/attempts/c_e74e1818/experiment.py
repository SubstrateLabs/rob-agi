from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_e74e1818.main import solve_e74e1818, identify_shapes, transform_shapes, reconstruct_grid

def print_grid(grid):
    for row in grid:
        print(''.join(str(cell) for cell in row))
    print()

def run_experiment():
    # Example grid from test case 0
    input_grid = ColoredGrid(values=[
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 4, 0, 4, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    print("Input Grid:")
    print_grid(input_grid.values)

    shapes = identify_shapes(input_grid)
    print(f"Number of shapes identified: {len(shapes)}")
    for i, shape in enumerate(shapes):
        print(f"Shape {i + 1}: {shape}")

    transformed_shapes = transform_shapes(shapes, input_grid.num_rows)
    print("\nTransformed Shapes:")
    for i, shape in enumerate(transformed_shapes):
        print(f"Shape {i + 1}: {shape}")

    output_grid = reconstruct_grid(transformed_shapes, input_grid)
    print("\nOutput Grid:")
    print_grid(output_grid.values)

    print("\nExpected Output Grid:")
    expected_output = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 4, 0, 4, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    print_grid(expected_output)

if __name__ == "__main__":
    run_experiment()
