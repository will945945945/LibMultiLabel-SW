#!/bin/bash

# Check arguments
if [ $# -ne 2 ]; then
  echo "Usage: $0 <input_file> <percentage>"
  echo "Example: $0 input.txt 70"
  exit 1
fi

input_file=$1
output_file1=$1.trva
output_file2=$1.te
percentage=$2

# Ensure the percentage is valid
if ! [[ $percentage =~ ^[0-9]+$ ]] || [ $percentage -lt 0 ] || [ $percentage -gt 100 ]; then
  echo "Error: Percentage must be an integer between 0 and 100."
  exit 1
fi

# Set a random seed for reproducibility
seed=42  # You can change this seed or make it a parameter if needed
RANDOM=$seed

# Split the file
awk -v perc="$percentage" -v seed="$seed" 'BEGIN {srand(seed)} {if (rand() * 100 < perc) print > "'"$output_file1"'" ; else print > "'"$output_file2"'"}' "$input_file"

echo "File successfully split into:"
echo "  - $output_file1 (contains ~$percentage% of lines)"
echo "  - $output_file2 (contains ~$((100 - percentage))% of lines)"

