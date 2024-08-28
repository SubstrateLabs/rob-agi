from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_e99362f0.main import solve_e99362f0
from collections import Counter
import json

def analyze_color_distribution(grid: ColoredGrid):
    flat_grid = [color for row in grid.values for color in row]
    return dict(Counter(flat_grid))

def compare_distributions(input_grid: ColoredGrid, output_grid: ColoredGrid):
    input_dist = analyze_color_distribution(input_grid)
    output_dist = analyze_color_distribution(output_grid)
    
    input_total = sum(input_dist.values())
    output_total = sum(output_dist.values())
    
    input_percentages = {color: count / input_total * 100 for color, count in input_dist.items()}
    output_percentages = {color: count / output_total * 100 for color, count in output_dist.items()}
    
    return input_percentages, output_percentages

def run_experiment():
    results = {}
    for i in range(6):
        input_grid = ColoredGrid.from_file(f"rob_agi/attempts/c_e99362f0/example_{i}_input.txt")
        expected_output = ColoredGrid.from_file(f"rob_agi/attempts/c_e99362f0/example_{i}_output.txt")
        actual_output = solve_e99362f0(input_grid)
        
        input_dist, expected_dist = compare_distributions(input_grid, expected_output)
        _, actual_dist = compare_distributions(input_grid, actual_output)
        
        results[f"example_{i}"] = {
            "input_distribution": input_dist,
            "expected_output_distribution": expected_dist,
            "actual_output_distribution": actual_dist
        }
    
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_experiment()
