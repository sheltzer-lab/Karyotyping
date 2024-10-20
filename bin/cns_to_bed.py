#!/usr/bin/env python

import sys

# Input and output file paths
input_file = sys.argv[1]
output_file = sys.argv[2]

# Open input and output files
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    # Skip header line if present (assuming the first line is a header)
    next(infile)
    
    # Iterate through each line in the input file
    for line in infile:
        # Split the line into fields based on tabs
        fields = line.strip().split('\t')
        
        # Extract relevant columns
        chromosome = fields[0]
        start = int(fields[1])
        end = int(fields[2])
        gene = str(fields[3])
        depth = float(fields[4])
        log2 = float(fields[5])
        weight = float(fields[6])
        
        # Write BED format to output file
        bed_line = f'{chromosome}\t{start}\t{end}\t{gene}\t{depth}\t{log2}\t{weight}\n'
        outfile.write(bed_line)
