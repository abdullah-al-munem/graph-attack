import json
from collections import defaultdict

# Define all results from the tables
results = {
    "GCN": {
        "Cora": {
            "Random": [0.18, 0.29, 0.28, 0.31, 0.33],
            "FGA": [0.26, 0.38, 0.41, 0.49, 0.51],
            "Nettack": [0.33, 0.45, 0.56, 0.60, 0.61],
            "SGAttack": [0.29, 0.39, 0.49, 0.54, 0.52],
            "GOttack": [0.41, 0.54, 0.62, 0.66, 0.71],
            "GAEttack": [0.30, 0.43, 0.50, 0.58, 0.63]
        },
        "Citeseer": {
            "Random": [0.20, 0.23, 0.26, 0.34, 0.31],
            "FGA": [0.23, 0.36, 0.40, 0.42, 0.50],
            "Nettack": [0.26, 0.45, 0.53, 0.58, 0.59],
            "SGAttack": [0.23, 0.35, 0.40, 0.50, 0.56],
            "GOttack": [0.46, 0.63, 0.72, 0.76, 0.78],
            "GAEttack": [0.24, 0.38, 0.50, 0.59, 0.65]
        },
        "Polblogs": {
            "Random": [0.15, 0.23, 0.32, 0.33, 0.37],
            "FGA": [0.10, 0.17, 0.17, 0.25, 0.23],
            "Nettack": [0.14, 0.27, 0.32, 0.34, 0.36],
            "SGAttack": [0.13, 0.17, 0.24, 0.32, 0.32],
            "GOttack": [0.41, 0.46, 0.51, 0.52, 0.55],
            "GAEttack": [0.11, 0.28, 0.34, 0.40, 0.45]
        },
        "BlogCatalog": {
            "Random": [0.14, 0.13, 0.17, 0.16, 0.18],
            "FGA": [0.16, 0.22, 0.24, 0.26, 0.33],
            "Nettack": [0.21, 0.31, 0.34, 0.39, 0.46],
            "SGAttack": [0.20, 0.27, 0.33, 0.39, 0.41],
            "GOttack": [0.22, 0.25, 0.35, 0.37, 0.45],
            "GAEttack": [0.18, 0.28, 0.35, 0.40, 0.41]
        }
    },
    "GIN": {
        "Cora": {
            "Random": [0.41, 0.47, 0.44, 0.46, 0.46],
            "FGA": [0.39, 0.46, 0.54, 0.49, 0.53],
            "Nettack": [0.47, 0.48, 0.53, 0.59, 0.65],
            "SGAttack": [0.47, 0.41, 0.51, 0.55, 0.58],
            "GOttack": [0.37, 0.48, 0.54, 0.59, 0.64],
            "GAEttack": [0.42, 0.61, 0.74, 0.78, 0.80]
        },
        "Citeseer": {
            "Random": [0.39, 0.44, 0.52, 0.51, 0.52],
            "FGA": [0.38, 0.47, 0.49, 0.50, 0.52],
            "Nettack": [0.39, 0.48, 0.58, 0.62, 0.63],
            "SGAttack": [0.40, 0.44, 0.47, 0.49, 0.54],
            "GOttack": [0.57, 0.60, 0.66, 0.74, 0.76],
            "GAEttack": [0.41, 0.64, 0.74, 0.78, 0.82]
        },
        "Polblogs": {
            "Random": [0.12, 0.21, 0.23, 0.25, 0.27],
            "FGA": [0.12, 0.10, 0.12, 0.12, 0.14],
            "Nettack": [0.13, 0.14, 0.18, 0.21, 0.29],
            "SGAttack": [0.08, 0.13, 0.14, 0.14, 0.18],
            "GOttack": [0.15, 0.23, 0.28, 0.32, 0.34],
            "GAEttack": [0.11, 0.16, 0.22, 0.28, 0.34]
        },
        "BlogCatalog": {
            "Random": [0.78, 0.76, 0.76, 0.76, 0.75],
            "FGA": [0.77, 0.78, 0.75, 0.75, 0.79],
            "Nettack": [0.80, 0.73, 0.81, 0.79, 0.82],
            "SGAttack": [0.75, 0.81, 0.77, 0.81, 0.80],
            "GOttack": [0.64, 0.66, 0.61, 0.65, 0.66],
            "GAEttack": [0.79, 0.97, 0.99, 1.00, 1.00]
        }
    },
    "GraphSage": {
        "Cora": {
            "Random": [0.51, 0.53, 0.59, 0.57, 0.62],
            "FGA": [0.54, 0.57, 0.61, 0.60, 0.64],
            "Nettack": [0.53, 0.61, 0.70, 0.67, 0.67],
            "SGAttack": [0.56, 0.62, 0.66, 0.66, 0.65],
            "GOttack": [0.59, 0.78, 0.86, 0.88, 0.92],
            "GAEttack": [0.57, 0.80, 0.89, 0.94, 0.96]
        },
        "Citeseer": {
            "Random": [0.54, 0.60, 0.65, 0.59, 0.63],
            "FGA": [0.58, 0.53, 0.60, 0.68, 0.64],
            "Nettack": [0.59, 0.58, 0.67, 0.71, 0.71],
            "SGAttack": [0.58, 0.55, 0.59, 0.62, 0.66],
            "GOttack": [0.61, 0.83, 0.92, 0.95, 0.97],
            "GAEttack": [0.60, 0.78, 0.92, 0.96, 0.96]
        },
        "Polblogs": {
            "Random": [0.24, 0.30, 0.31, 0.37, 0.43],
            "FGA": [0.18, 0.24, 0.24, 0.29, 0.29],
            "Nettack": [0.23, 0.30, 0.34, 0.40, 0.37],
            "SGAttack": [0.22, 0.25, 0.30, 0.30, 0.32],
            "GOttack": [0.29, 0.36, 0.44, 0.49, 0.54],
            "GAEttack": [0.24, 0.37, 0.50, 0.61, 0.64]
        },
        "BlogCatalog": {
            "Random": [0.44, 0.41, 0.41, 0.44, 0.42],
            "FGA": [0.40, 0.40, 0.40, 0.44, 0.43],
            "Nettack": [0.44, 0.45, 0.43, 0.47, 0.43],
            "SGAttack": [0.42, 0.43, 0.44, 0.47, 0.48],
            "GOttack": [0.00, 0.00, 0.00, 0.00, 0.00],  # GOttack missing in table
            "GAEttack": [0.37, 0.57, 0.68, 0.77, 0.81]
        }
    },
    "RobustGCN": {
        "Cora": {
            "Random": [0.29, 0.41, 0.46, 0.49, 0.55],
            "FGA": [0.29, 0.38, 0.52, 0.58, 0.62],
            "Nettack": [0.38, 0.51, 0.65, 0.71, 0.73],
            "SGAttack": [0.39, 0.56, 0.68, 0.72, 0.75],
            "GOttack": [0.43, 0.61, 0.70, 0.69, 0.74],
            "GAEttack": [0.28, 0.52, 0.61, 0.67, 0.75]
        },
        "Citeseer": {
            "Random": [0.48, 0.52, 0.59, 0.59, 0.65],
            "FGA": [0.33, 0.49, 0.55, 0.59, 0.69],
            "Nettack": [0.40, 0.57, 0.63, 0.68, 0.74],
            "SGAttack": [0.52, 0.63, 0.68, 0.71, 0.76],
            "GOttack": [0.48, 0.59, 0.72, 0.74, 0.76],
            "GAEttack": [0.37, 0.56, 0.68, 0.74, 0.76]
        },
        "Polblogs": {
            "Random": [0.28, 0.39, 0.45, 0.48, 0.50],
            "FGA": [0.21, 0.31, 0.39, 0.41, 0.44],
            "Nettack": [0.30, 0.39, 0.45, 0.49, 0.50],
            "SGAttack": [0.44, 0.52, 0.55, 0.57, 0.60],
            "GOttack": [0.40, 0.40, 0.46, 0.51, 0.50],
            "GAEttack": [0.36, 0.48, 0.50, 0.52, 0.54]
        },
        "BlogCatalog": {
            "Random": [0.24, 0.23, 0.24, 0.26, 0.23],
            "FGA": [0.25, 0.28, 0.28, 0.28, 0.29],
            "Nettack": [0.24, 0.29, 0.29, 0.35, 0.38],
            "SGAttack": [0.29, 0.29, 0.28, 0.33, 0.37],
            "GOttack": [0.36, 0.31, 0.30, 0.35, 0.36],
            "GAEttack": [0.24, 0.33, 0.39, 0.43, 0.46]
        }
    },
    "GCN-Jaccard": {
        "Cora": {
            "Random": [0.23, 0.29, 0.34, 0.39, 0.39],
            "FGA": [0.30, 0.37, 0.44, 0.50, 0.52],
            "Nettack": [0.30, 0.43, 0.48, 0.56, 0.59],
            "SGAttack": [0.30, 0.41, 0.47, 0.50, 0.54],
            "GOttack": [0.39, 0.46, 0.61, 0.62, 0.58],
            "GAEttack": [0.29, 0.44, 0.51, 0.60, 0.63]
        },
        "Citeseer": {
            "Random": [0.26, 0.37, 0.42, 0.46, 0.48],
            "FGA": [0.25, 0.37, 0.39, 0.49, 0.53],
            "Nettack": [0.32, 0.49, 0.56, 0.60, 0.65],
            "SGAttack": [0.35, 0.52, 0.58, 0.64, 0.71],
            "GOttack": [0.42, 0.53, 0.58, 0.62, 0.73],
            "GAEttack": [0.32, 0.48, 0.56, 0.61, 0.68]
        },
        "Polblogs": {
            "Random": [0.49, 0.47, 0.45, 0.55, 0.45],
            "FGA": [0.49, 0.53, 0.50, 0.53, 0.44],
            "Nettack": [0.49, 0.50, 0.53, 0.48, 0.51],
            "SGAttack": [0.48, 0.50, 0.53, 0.49, 0.51],
            "GOttack": [0.53, 0.43, 0.49, 0.46, 0.54],
            "GAEttack": [0.49, 0.66, 0.77, 0.85, 0.89]
        },
        "BlogCatalog": {
            "Random": [0.11, 0.13, 0.14, 0.14, 0.16],
            "FGA": [0.19, 0.23, 0.27, 0.27, 0.36],
            "Nettack": [0.24, 0.30, 0.34, 0.38, 0.41],
            "SGAttack": [0.22, 0.26, 0.30, 0.37, 0.37],
            "GOttack": [0.20, 0.30, 0.36, 0.43, 0.42],
            "GAEttack": [0.20, 0.28, 0.34, 0.37, 0.38]
        }
    },
    "GCN-SVD": {
        "Cora": {
            "Random": [0.25, 0.33, 0.34, 0.39, 0.44],
            "FGA": [0.26, 0.25, 0.24, 0.26, 0.26],
            "Nettack": [0.27, 0.29, 0.26, 0.30, 0.32],
            "SGAttack": [0.24, 0.27, 0.29, 0.29, 0.28],
            "GOttack": [0.28, 0.26, 0.30, 0.27, 0.26],
            "GAEttack": [0.26, 0.38, 0.45, 0.49, 0.52]
        },
        "Citeseer": {
            "Random": [0.33, 0.34, 0.40, 0.39, 0.45],
            "FGA": [0.17, 0.20, 0.21, 0.21, 0.22],
            "Nettack": [0.18, 0.25, 0.24, 0.25, 0.30],
            "SGAttack": [0.21, 0.22, 0.22, 0.25, 0.28],
            "GOttack": [0.25, 0.28, 0.27, 0.31, 0.38],
            "GAEttack": [0.24, 0.37, 0.45, 0.47, 0.52]
        },
        "Polblogs": {
            "Random": [0.22, 0.28, 0.33, 0.33, 0.38],
            "FGA": [0.14, 0.16, 0.15, 0.17, 0.14],
            "Nettack": [0.16, 0.18, 0.21, 0.29, 0.34],
            "SGAttack": [0.11, 0.13, 0.16, 0.16, 0.16],
            "GOttack": [0.10, 0.13, 0.12, 0.16, 0.16],
            "GAEttack": [0.15, 0.18, 0.24, 0.29, 0.32]
        },
        "BlogCatalog": {
            "Random": [0.15, 0.14, 0.18, 0.14, 0.17],
            "FGA": [0.14, 0.13, 0.20, 0.20, 0.20],
            "Nettack": [0.18, 0.20, 0.20, 0.26, 0.27],
            "SGAttack": [0.17, 0.20, 0.21, 0.20, 0.24],
            "GOttack": [0.20, 0.23, 0.28, 0.20, 0.35],
            "GAEttack": [0.20, 0.30, 0.37, 0.40, 0.47]
        }
    }
}

# Save to JSON
with open('gaettack_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("JSON file 'gaettack_results.json' created successfully!\n")

# Function to rank attacks for a single task (dense ranking with ties)
def rank_attacks(scores_dict):
    """
    Ranks attacks by score (higher is better).
    Returns dict: {attack_name: rank} where rank is 1 (highest), 2 (2nd), 3 (3rd), etc.
    Uses dense ranking: ties get same rank, next rank continues sequentially.
    """
    sorted_attacks = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    rankings = {}
    current_rank = 1
    prev_score = None
    
    for attack, score in sorted_attacks:
        if prev_score is not None and score < prev_score:
            current_rank += 1
        rankings[attack] = current_rank
        prev_score = score
    
    return rankings

# Initialize counters
all_attacks = ["Random", "FGA", "Nettack", "SGAttack", "GOttack", "GAEttack"]
overall_stats = {attack: {"highest": 0, "2nd": 0, "3rd": 0} for attack in all_attacks}
per_dataset_stats = defaultdict(lambda: {attack: {"highest": 0, "2nd": 0, "3rd": 0} for attack in all_attacks})
per_model_stats = defaultdict(lambda: {attack: {"highest": 0, "2nd": 0, "3rd": 0} for attack in all_attacks})
per_model_dataset_stats = defaultdict(lambda: defaultdict(lambda: {attack: {"highest": 0, "2nd": 0, "3rd": 0} for attack in all_attacks}))

# Process all tasks
total_tasks = 0
for model_name, datasets in results.items():
    for dataset_name, attacks in datasets.items():
        for budget_idx in range(5):  # 5 budgets
            total_tasks += 1
            
            # Get scores for this task
            scores = {}
            for attack_name, values in attacks.items():
                # Skip GOttack for GraphSage-BlogCatalog (missing data)
                if model_name == "GraphSage" and dataset_name == "BlogCatalog" and attack_name == "GOttack":
                    continue
                scores[attack_name] = values[budget_idx]
            
            # Rank attacks
            rankings = rank_attacks(scores)
            
            # Update statistics
            for attack_name, rank in rankings.items():
                if rank == 1:
                    overall_stats[attack_name]["highest"] += 1
                    per_dataset_stats[dataset_name][attack_name]["highest"] += 1
                    per_model_stats[model_name][attack_name]["highest"] += 1
                    per_model_dataset_stats[model_name][dataset_name][attack_name]["highest"] += 1
                elif rank == 2:
                    overall_stats[attack_name]["2nd"] += 1
                    per_dataset_stats[dataset_name][attack_name]["2nd"] += 1
                    per_model_stats[model_name][attack_name]["2nd"] += 1
                    per_model_dataset_stats[model_name][dataset_name][attack_name]["2nd"] += 1
                elif rank == 3:
                    overall_stats[attack_name]["3rd"] += 1
                    per_dataset_stats[dataset_name][attack_name]["3rd"] += 1
                    per_model_stats[model_name][attack_name]["3rd"] += 1
                    per_model_dataset_stats[model_name][dataset_name][attack_name]["3rd"] += 1

# Generate comprehensive report
report = []
report.append("=" * 80)
report.append("COMPREHENSIVE GAETTACK EXPERIMENTAL RESULTS")
report.append("=" * 80)
report.append("")

# Overall Statistics
report.append("=" * 80)
report.append("1. OVERALL STATISTICS (Total Tasks: 120)")
report.append("=" * 80)
report.append("")
report.append(f"{'Attack':<15} {'Highest':<20} {'2nd Highest':<20} {'3rd Highest':<20}")
report.append("-" * 80)
for attack in all_attacks:
    stats = overall_stats[attack]
    report.append(f"{attack:<15} {stats['highest']:>3} out of 120 tasks  {stats['2nd']:>3} out of 120 tasks  {stats['3rd']:>3} out of 120 tasks")
report.append("")

# Per-Dataset Statistics
report.append("=" * 80)
report.append("2. PER-DATASET BREAKDOWN")
report.append("=" * 80)
report.append("")

datasets_list = ["Cora", "Citeseer", "Polblogs", "BlogCatalog"]
for dataset in datasets_list:
    report.append(f"\n{'-' * 80}")
    report.append(f"Dataset: {dataset} (30 tasks: 6 models × 5 budgets)")
    report.append(f"{'-' * 80}")
    report.append(f"{'Attack':<15} {'Highest':<20} {'2nd Highest':<20} {'3rd Highest':<20}")
    report.append("-" * 80)
    for attack in all_attacks:
        stats = per_dataset_stats[dataset][attack]
        report.append(f"{attack:<15} {stats['highest']:>2} out of 30 tasks   {stats['2nd']:>2} out of 30 tasks   {stats['3rd']:>2} out of 30 tasks")
    report.append("")

# Per-Model Statistics
report.append("=" * 80)
report.append("3. PER-MODEL BREAKDOWN")
report.append("=" * 80)
report.append("")

models_list = ["GCN", "GIN", "GraphSage", "RobustGCN", "GCN-Jaccard", "GCN-SVD"]
for model in models_list:
    # Adjust task count for GraphSage (missing GOttack for BlogCatalog)
    task_count = 19 if model == "GraphSage" else 20
    report.append(f"\n{'-' * 80}")
    report.append(f"Model: {model} ({task_count} tasks: 4 datasets × 5 budgets)")
    report.append(f"{'-' * 80}")
    report.append(f"{'Attack':<15} {'Highest':<20} {'2nd Highest':<20} {'3rd Highest':<20}")
    report.append("-" * 80)
    for attack in all_attacks:
        stats = per_model_stats[model][attack]
        report.append(f"{attack:<15} {stats['highest']:>2} out of {task_count} tasks   {stats['2nd']:>2} out of {task_count} tasks   {stats['3rd']:>2} out of {task_count} tasks")
    report.append("")

# Per-Model Per-Dataset Statistics
report.append("=" * 80)
report.append("4. PER-MODEL PER-DATASET BREAKDOWN")
report.append("=" * 80)
report.append("")

for model in models_list:
    report.append(f"\n{'=' * 80}")
    report.append(f"MODEL: {model}")
    report.append(f"{'=' * 80}")
    
    for dataset in datasets_list:
        # Adjust task count for GraphSage-BlogCatalog
        task_count = 5 if not (model == "GraphSage" and dataset == "BlogCatalog") else 5
        # Check if GOttack is missing
        attacks_in_task = all_attacks if not (model == "GraphSage" and dataset == "BlogCatalog") else [a for a in all_attacks if a != "GOttack"]
        
        report.append(f"\n{'-' * 80}")
        report.append(f"{model} on {dataset} (5 tasks: 5 budgets)")
        report.append(f"{'-' * 80}")
        report.append(f"{'Attack':<15} {'Highest':<20} {'2nd Highest':<20} {'3rd Highest':<20}")
        report.append("-" * 80)
        for attack in attacks_in_task:
            stats = per_model_dataset_stats[model][dataset][attack]
            report.append(f"{attack:<15} {stats['highest']:>1} out of 5 tasks    {stats['2nd']:>1} out of 5 tasks    {stats['3rd']:>1} out of 5 tasks")
        report.append("")

# Write report to file
report_text = "\n".join(report)
with open('gaettack_comprehensive_report.txt', 'w') as f:
    f.write(report_text)

print(report_text)
print("\nReport saved to 'gaettack_comprehensive_report.txt'")

# Summary for GAEttack
print("\n" + "=" * 80)
print("KEY FINDINGS FOR GAETTACK:")
print("=" * 80)
print(f"Overall Performance:")
print(f"  - Highest: {overall_stats['GAEttack']['highest']} out of 120 tasks")
print(f"  - 2nd Highest: {overall_stats['GAEttack']['2nd']} out of 120 tasks")
print(f"  - 3rd Highest: {overall_stats['GAEttack']['3rd']} out of 120 tasks")
print(f"  - Top-2 Combined: {overall_stats['GAEttack']['highest'] + overall_stats['GAEttack']['2nd']} out of 120 tasks ({(overall_stats['GAEttack']['highest'] + overall_stats['GAEttack']['2nd'])/120*100:.1f}%)")
