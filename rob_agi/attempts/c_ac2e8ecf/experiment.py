from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_ac2e8ecf.main import solve_ac2e8ecf

def run_experiment():
    # Create a sample input grid
    input_grid = ColoredGrid(values=[
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1],
        [0, 2, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0],
        [2, 2, 2, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
        [0, 0, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 5, 0, 0, 5, 0, 0, 0, 0, 2, 2, 2],
        [0, 0, 5, 5, 5, 5, 0, 0, 0, 0, 2, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    # Solve the puzzle
    output_grid = solve_ac2e8ecf(input_grid)

    # Print the input and output grids
    print("Input Grid:")
    print(input_grid)
    print("\nOutput Grid:")
    print(output_grid)

    # Analyze the output
    analyze_output(input_grid, output_grid)

def analyze_output(input_grid: ColoredGrid, output_grid: ColoredGrid):
    input_shapes = analyze_shapes(input_grid)
    output_shapes = analyze_shapes(output_grid)

    print("\nAnalysis:")
    print(f"Number of shapes in input: {len(input_shapes)}")
    print(f"Number of shapes in output: {len(output_shapes)}")

    for color in range(1, 10):
        input_color_shapes = [s for s in input_shapes if s['color'] == color]
        output_color_shapes = [s for s in output_shapes if s['color'] == color]
        if input_color_shapes:
            print(f"\nColor {color}:")
            print(f"  Input shapes: {len(input_color_shapes)}")
            print(f"  Output shapes: {len(output_color_shapes)}")
            for i, shape in enumerate(input_color_shapes):
                if i < len(output_color_shapes):
                    input_centroid = shape['centroid']
                    output_centroid = output_color_shapes[i]['centroid']
                    print(f"  Shape {i+1}: Input centroid: {input_centroid}, Output centroid: {output_centroid}")

def analyze_shapes(grid: ColoredGrid):
    shapes = []
    for color in range(1, 10):  # Exclude black (0)
        regions = grid.find_connected_regions(color)
        for region in regions:
            shape = {
                'color': color,
                'size': len(region),
                'bounding_box': get_bounding_box(region),
                'cells': region,
                'centroid': get_centroid(region),
            }
            shapes.append(shape)
    return shapes

def get_bounding_box(region):
    min_row = min(r for r, _ in region)
    max_row = max(r for r, _ in region)
    min_col = min(c for _, c in region)
    max_col = max(c for _, c in region)
    return (min_row, min_col, max_row - min_row + 1, max_col - min_col + 1)

def get_centroid(region):
    avg_row = sum(r for r, _ in region) / len(region)
    avg_col = sum(c for _, c in region) / len(region)
    return (avg_row, avg_col)

if __name__ == "__main__":
    run_experiment()
