import pandas as pd

def extract_codes(input_file, output_file):
    # Load TSV
    df = pd.read_csv(input_file, sep='\t', dtype=str)

    # Use a list to preserve order
    code_lines = []

    for _, row in df.iterrows():
        codes = []

        # Add the single code
        if pd.notna(row['code']):
            codes.append(row['code'])

        # Avoid empty lines
        if codes:
            code_lines.append('|'.join(codes))

    # Write to output file
    with open(output_file, 'w') as f:
        for line in code_lines:
            f.write(line + '\n')

    print(f"Wrote {len(code_lines)} lines to {output_file}")

# Example usage
extract_codes("_REEL/bc5cdr_Disease_medic/xlinker_preds.tsv", "labels_bc5cdr_disease_medic.txt")

