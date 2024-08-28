from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_2697da3f.main import solve_2697da3f, extract_core_pattern, scale_core_pattern

def analyze_scaling(input_grid):
    core_pattern = extract_core_pattern(input_grid)
    max_dim = max(input_grid.get_dimensions())
    output_size = 2 * max_dim - 1
    scaled_core = scale_core_pattern(core_pattern, output_size // 2)
    
    print(f"Input dimensions: {input_grid.get_dimensions()}")
    print(f"Core pattern dimensions: {len(core_pattern)}x{len(core_pattern[0])}")
    print(f"Scaled core dimensions: {len(scaled_core)}x{len(scaled_core[0])}")
    print(f"Output size: {output_size}x{output_size}")
    
    input_density = sum(sum(row) != 0 for row in input_grid.values) / (input_grid.get_dimensions()[0] * input_grid.get_dimensions()[1])
    scaled_density = sum(sum(row) != 0 for row in scaled_core) / (len(scaled_core) * len(scaled_core[0]))
    
    print(f"Input pattern density: {input_density:.2f}")
    print(f"Scaled pattern density: {scaled_density:.2f}")

# Test with example inputs
example_inputs = [
    [[0, 0, 0, 0, 0, 0, 0],
     [0, 4, 4, 4, 0, 4, 0],
     [0, 0, 0, 4, 4, 4, 0],
     [0, 0, 0, 0, 4, 0, 0],
     [0, 0, 0, 4, 4, 4, 0],
     [0, 4, 4, 4, 0, 4, 0],
     [0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0]],
    
    [[0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0],
     [0, 0, 4, 4, 0, 0, 0],
     [0, 4, 0, 4, 4, 0, 0],
     [0, 0, 4, 4, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0]]
]

for i, input_grid in enumerate(example_inputs):
    print(f"\nAnalyzing Example {i + 1}")
    analyze_scaling(ColoredGrid(values=input_grid))
