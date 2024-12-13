#!/bin/bash

# Prompt for the folder path
read -p "Enter the folder path (e.g., C:/Projects/github): " folder_path

# Change to the specified directory
cd "$folder_path" || { echo "Directory not found: $folder_path"; exit 1; }

# Loop through each directory
for dir in */; do
    cd "$dir" || continue  # Change to the directory, skip if it fails
    pwd

    # Get the remote repository URL
    git config --get remote.origin.url

    # Stage and commit changes
    git add .
    git commit -m "bash push all"

    # Push changes to the remote repository
    git push

    # Return to the original directory
    cd .. || exit
    echo "----------------------------------------------------------------"
done
