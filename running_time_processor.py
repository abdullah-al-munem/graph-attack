import json
import os
import re
from collections import defaultdict

def parse_time_string(time_str):
    """
    Parse time string like "0 minutes, 1.1219561100006104 seconds" to float seconds
    """
    # Extract minutes and seconds using regex
    pattern = r'(\d+)\s*minutes?,\s*([0-9.]+)\s*seconds?'
    match = re.search(pattern, time_str)
    
    if match:
        minutes = float(match.group(1))
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    else:
        # If pattern doesn't match, try to extract just seconds
        seconds_pattern = r'([0-9.]+)\s*seconds?'
        seconds_match = re.search(seconds_pattern, time_str)
        if seconds_match:
            return float(seconds_match.group(1))
    
    return 0.0

def process_running_time_files(folder_path="running_time_result"):
    """
    Process all JSON files in the folder and create the target JSON structure
    """
    # Define the models and datasets
    attack_models = ['Proposed_model', 'Random_attack', 'FGA', 'Nettack', 'SGAttack_attack']
    datasets = ['cora', 'citeseer', 'polblogs']
    
    # Dictionary to store all data: {attack_model: {dataset: {budget: [times]}}}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Process all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            # Parse filename to extract components
            # Format: running_time_{iteration}_dataset_{dataset}_defense_{defense}.json
            pattern = r'running_time_(\d+)_dataset_([^_]+)_defense_([^.]+)\.json'
            match = re.match(pattern, filename)
            
            if match:
                iteration = int(match.group(1))
                dataset = match.group(2)
                defense = match.group(3)
                
                # Only process GCN defense files
                if defense == 'gcn' and dataset in datasets:
                    filepath = os.path.join(folder_path, filename)
                    
                    try:
                        with open(filepath, 'r') as f:
                            json_data = json.load(f)
                        
                        # Process each attack model
                        for attack_model in attack_models:
                            if attack_model in json_data:
                                # Process each budget entry
                                for entry in json_data[attack_model]:
                                    budget = entry['budget']
                                    running_time_str = entry['running_time']
                                    
                                    # Convert time string to float
                                    time_seconds = parse_time_string(running_time_str)
                                    
                                    # Store the time
                                    data[attack_model][dataset][budget].append(time_seconds)
                    
                    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                        print(f"Error processing {filename}: {e}")
    
    # Calculate averages and create target JSON structure
    target_json = {}
    
    for attack_model in attack_models:
        target_json[attack_model] = {}
        
        for dataset in datasets:
            target_json[attack_model][dataset] = []
            
            # Process budgets 1 to 7
            for budget in range(1, 8):
                if budget in data[attack_model][dataset]:
                    times = data[attack_model][dataset][budget]
                    if times:  # If we have data for this budget
                        avg_time = sum(times) / len(times)
                        target_json[attack_model][dataset].append({
                            "avg_running_time_of_5": round(avg_time, 6),
                            "budget": budget
                        })
                    else:
                        # No data for this budget
                        target_json[attack_model][dataset].append({
                            "avg_running_time_of_5": 0.0,
                            "budget": budget
                        })
                else:
                    # No data for this budget
                    target_json[attack_model][dataset].append({
                        "avg_running_time_of_5": 0.0,
                        "budget": budget
                    })
    
    return target_json

def main():
    # Process the files
    result = process_running_time_files()
    
    # Save to output file
    output_filename = "averaged_running_times_gcn.json"
    with open(output_filename, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to {output_filename}")
    
    # Print summary statistics
    print("\nSummary:")
    for attack_model in result:
        print(f"\n{attack_model}:")
        for dataset in result[attack_model]:
            print(f"  {dataset}: {len(result[attack_model][dataset])} budget entries")
            
            # Show sample data for budget 1
            if result[attack_model][dataset]:
                budget_1_data = next((item for item in result[attack_model][dataset] if item["budget"] == 1), None)
                if budget_1_data:
                    print(f"    Budget 1 avg time: {budget_1_data['avg_running_time_of_5']} seconds")

if __name__ == "__main__":
    main()
