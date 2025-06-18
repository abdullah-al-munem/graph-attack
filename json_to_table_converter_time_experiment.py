import json
import pandas as pd
import numpy as np

def create_comprehensive_table(json_file="averaged_running_times_gcn_with_std.json"):
    """
    Read the JSON file and create a comprehensive table showing all attack models,
    datasets, and budgets in a single, easy-to-compare format.
    Dataset-first organization for better visual comparison.
    Now shows avg ± std format.
    """
    
    # Read the JSON file
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_file} not found. Make sure the file exists.")
        return None
    
    # Extract attack models and datasets
    attack_models = list(data.keys())
    datasets = list(data[attack_models[0]].keys()) if attack_models else []
    
    # Sort datasets and attack models for consistent ordering
    datasets = sorted(datasets)
    attack_models = sorted(attack_models)
    
    # Create a list to store all rows
    rows = []
    
    # Process each dataset first, then all attack models for that dataset
    for dataset in datasets:
        for attack_model in attack_models:
            if attack_model in data and dataset in data[attack_model]:
                # Create a row for this dataset + attack_model combination
                row = {
                    'Dataset': dataset,
                    'Attack_Model': attack_model
                }
                
                # Add budget columns (Budget_1 to Budget_7)
                budget_data = data[attack_model][dataset]
                
                # Initialize all budgets to default
                for budget_num in range(1, 8):
                    row[f'Budget_{budget_num}'] = "0.0 ± 0.0"
                
                # Fill in actual data
                for entry in budget_data:
                    budget_num = entry['budget']
                    # Use avg_plus_minus_std if available, otherwise construct it
                    if 'avg_plus_minus_std' in entry:
                        avg_plus_minus_std = entry['avg_plus_minus_std']
                    else:
                        # Fallback: construct from individual values
                        avg_time = entry.get('avg_running_time_of_5', 0.0)
                        std_dev = entry.get('std_deviation', 0.0)
                        avg_plus_minus_std = f"{round(avg_time, 6)} ± {round(std_dev, 6)}"
                    
                    row[f'Budget_{budget_num}'] = avg_plus_minus_std
                
                rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Reorder columns for better readability (Dataset first, then Attack_Model)
    base_columns = ['Dataset', 'Attack_Model']
    budget_columns = [f'Budget_{i}' for i in range(1, 8)]
    df = df[base_columns + budget_columns]
    
    # Data is already sorted by dataset first, then attack model
    df = df.reset_index(drop=True)
    
    return df

def create_pivot_table(json_file="averaged_running_times_gcn_with_std.json"):
    """
    Create a pivot table format that's more compact and easier to compare
    Now shows avg ± std format.
    """
    
    # Read the JSON file
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_file} not found. Make sure the file exists.")
        return None
    
    # Create a list for long format data
    long_data = []
    
    for attack_model in data:
        for dataset in data[attack_model]:
            for entry in data[attack_model][dataset]:
                # Use avg_plus_minus_std if available, otherwise construct it
                if 'avg_plus_minus_std' in entry:
                    avg_plus_minus_std = entry['avg_plus_minus_std']
                else:
                    # Fallback: construct from individual values
                    avg_time = entry.get('avg_running_time_of_5', 0.0)
                    std_dev = entry.get('std_deviation', 0.0)
                    avg_plus_minus_std = f"{round(avg_time, 6)} ± {round(std_dev, 6)}"
                
                long_data.append({
                    'Attack_Model': attack_model,
                    'Dataset': dataset,
                    'Budget': entry['budget'],
                    'Avg_Plus_Minus_Std': avg_plus_minus_std
                })
    
    # Create DataFrame
    df_long = pd.DataFrame(long_data)
    
    # Create pivot table with multi-level columns
    pivot_df = df_long.pivot_table(
        index=['Attack_Model'],
        columns=['Dataset', 'Budget'],
        values='Avg_Plus_Minus_Std',
        fill_value="0.0 ± 0.0",
        aggfunc='first'  # Since we're dealing with strings now
    )
    
    return pivot_df

def extract_average_for_comparison(avg_plus_minus_str):
    """
    Extract just the average value from 'avg ± std' string for numerical comparison
    """
    try:
        # Split by ± and take the first part (average)
        avg_part = avg_plus_minus_str.split('±')[0].strip()
        return float(avg_part)
    except:
        return 0.0

def save_to_excel_and_csv(df, pivot_df, base_filename="running_time_comparison_with_std"):
    """
    Save both formats to Excel and CSV files
    """
    
    # Save comprehensive table
    excel_file = f"{base_filename}_comprehensive.xlsx"
    csv_file = f"{base_filename}_comprehensive.csv"
    
    df.to_excel(excel_file, index=False)
    df.to_csv(csv_file, index=False)
    
    print(f"Comprehensive table saved to:")
    print(f"  - {excel_file}")
    print(f"  - {csv_file}")
    
    # Save pivot table
    pivot_excel_file = f"{base_filename}_pivot.xlsx"
    pivot_csv_file = f"{base_filename}_pivot.csv"
    
    pivot_df.to_excel(pivot_excel_file)
    pivot_df.to_csv(pivot_csv_file)
    
    print(f"\nPivot table saved to:")
    print(f"  - {pivot_excel_file}")
    print(f"  - {pivot_csv_file}")

def main():
    print("Converting JSON to table format with standard deviation...")
    
    # Create comprehensive table
    df = create_comprehensive_table()
    
    if df is not None:
        print(f"\nComprehensive Table Preview:")
        print("=" * 120)
        print(df.head(10))
        print(f"\nTable shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Show summary statistics
        print(f"\nSummary:")
        print(f"Datasets: {df['Dataset'].unique().tolist()}")
        print(f"Attack Models: {df['Attack_Model'].unique().tolist()}")
        
        # Show which attack model is fastest for each dataset-budget combination
        print(f"\nFastest Attack Model by Dataset (Budget 1):")
        for dataset in df['Dataset'].unique():
            subset = df[df['Dataset'] == dataset]
            if not subset.empty:
                # Extract numerical values for comparison
                subset_with_nums = subset.copy()
                subset_with_nums['Budget_1_numeric'] = subset_with_nums['Budget_1'].apply(extract_average_for_comparison)
                
                fastest_idx = subset_with_nums['Budget_1_numeric'].idxmin()
                fastest_model = subset.loc[fastest_idx, 'Attack_Model']
                fastest_time = subset.loc[fastest_idx, 'Budget_1']
                print(f"  {dataset}: {fastest_model} ({fastest_time} seconds)")
        
        # Show dataset grouping preview
        print(f"\nDataset Grouping Preview:")
        for dataset in df['Dataset'].unique():
            subset = df[df['Dataset'] == dataset]
            print(f"  {dataset}: {len(subset)} attack models (rows {subset.index.min()}-{subset.index.max()})")
        
        # Create pivot table
        pivot_df = create_pivot_table()
        
        if pivot_df is not None:
            print(f"\n\nPivot Table Preview:")
            print("=" * 120)
            print(pivot_df)
            
            # Save both formats
            save_to_excel_and_csv(df, pivot_df)
            
            # Create a single recommended table (comprehensive format)
            recommended_file = "recommended_running_time_comparison_with_std.xlsx"
            
            # Add some formatting for better readability
            with pd.ExcelWriter(recommended_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Running Time with Std Dev', index=False)
                
                # Get the workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets['Running Time with Std Dev']
                
                # Auto-adjust column widths (make them wider for the ± format)
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    # Make columns wider to accommodate "avg ± std" format
                    adjusted_width = min(max_length + 3, 25)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
                # Add visual separation between datasets using background colors
                from openpyxl.styles import PatternFill
                
                # Create alternating colors for different datasets
                dataset_colors = ['E8F4FD', 'F0F8E8', 'FDF2E8']  # Light blue, light green, light orange
                current_dataset = None
                color_index = 0
                
                for row_num, (index, row) in enumerate(df.iterrows(), start=2):  # Start from row 2 (after header)
                    if row['Dataset'] != current_dataset:
                        current_dataset = row['Dataset']
                        color_index = (color_index) % len(dataset_colors)
                    
                    # Apply background color to the entire row
                    fill = PatternFill(start_color=dataset_colors[color_index], 
                                     end_color=dataset_colors[color_index], 
                                     fill_type="solid")
                    
                    for col_num in range(1, len(df.columns) + 1):
                        worksheet.cell(row=row_num, column=col_num).fill = fill
            
            print(f"\n*** RECOMMENDED FILE: {recommended_file} ***")
            print("This file contains the most readable format with average ± standard deviation!")
    
    else:
        print("Failed to create table. Please check if the JSON file exists.")

if __name__ == "__main__":
    main()