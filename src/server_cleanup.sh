#!/bin/bash

# Base directory containing all super rounds
BASE_DIR="$HOME/Federated-Learning/src/coordination/instance"

# List of super rounds to skip
SKIP_LIST=("super_round11" "super_round12" "super_round13" "super_round14" "super_round15" "super_round16" "super_round17" "super_round18""super_round19" "super_round20" "super_round21" "super_round22" "super_round23" "super_round24""super_round30" "super_round32")  # Add more if needed

echo "Starting cleanup..."

# Loop through all super_round directories
find "$BASE_DIR" -maxdepth 1 -type d -name "super*" | while read super_dir
do
  super_name=$(basename "$super_dir")
  
  # Check if this super round should be skipped
  if [[ " ${SKIP_LIST[@]} " =~ " ${super_name} " ]]; then
    echo "Skipping: $super_name"
    continue
  fi
  
  echo "Processing: $super_name"
  
  # Loop through all training_round directories
  find "$super_dir" -maxdepth 1 -type d -name "training_round*" | while read round_dir
  do
    models_dir="$round_dir/client_models"
    
    if [ -d "$models_dir" ]; then
      echo "  Cleaning client_models in: $round_dir"
      
      # Delete ONLY .pth files within client_models folder
      find "$models_dir" -type f -name "*.pth" -exec rm -f {} \;
      
    fi
    
  done
  
done

echo "Cleanup complete!"
