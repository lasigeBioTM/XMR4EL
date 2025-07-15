import pandas as pd
import ast

def extract_codes(input_file, output_file):
    # Load TSV
    df = pd.read_csv(input_file, sep='\t', dtype=str)

    # Use a set to collect unique codes
    all_codes = set()

    for _, row in df.iterrows():
        # Add single code
        if pd.notna(row['code']):
            all_codes.add(row['code'])

        # Add list of codes
        if pd.notna(row['codes']):
            try:
                codes_list = ast.literal_eval(row['codes'])
                all_codes.update(codes_list)
            except Exception as e:
                print(f"Error parsing codes in row {row['doc_id']}: {e}")

    # Write to output file
    with open(output_file, 'w') as f:
        for code in sorted(all_codes):
            f.write(code + '\n')

    print(f"Extracted {len(all_codes)} unique codes to {output_file}")

# Example usage
extract_codes("_REEL/bc5cdr_Disease_medic/xlinker_preds.tsv", "all_codes.txt")

