from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_009d5c81.main import analyze_shape, solve_009d5c81
import yaml

def load_examples():
    with open('rob_agi/attempts/c_009d5c81/visual_descriptions.yaml', 'r') as file:
        return yaml.safe_load(file)

def create_grid_from_description(description):
    lines = description.strip().split('\n')
    grid = [[0 for _ in range(14)] for _ in range(14)]
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char == '8':
                grid[i][j] = 8
            elif char == '1':
                grid[i][j] = 1
    return ColoredGrid(values=grid)

def analyze_examples():
    examples = load_examples()
    for name, example in examples.items():
        if name.startswith('example_'):
            print(f"\nAnalyzing {name}:")
            input_grid = create_grid_from_description(example['input'])
            shape = input_grid.find_connected_regions(8)[0]
            complexity, geometric_score, organic_score = analyze_shape(input_grid, shape)
            output = solve_009d5c81(input_grid)
            new_color = output.values[shape[0][0]][shape[0][1]]
            print(f"Complexity: {complexity:.2f}")
            print(f"Geometric Score: {geometric_score:.2f}")
            print(f"Organic Score: {organic_score:.2f}")
            print(f"New Color: {new_color}")

if __name__ == "__main__":
    analyze_examples()
