def clean_env_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Remove everything after '=' if it exists
            if '=' in line:
                key = line.split('=', 1)[0]  # Get everything before the '=' sign
                outfile.write(f"{key}=\n")  # Write key with '=' to output file
            else:
                outfile.write(line)  # Write line as is if no '=' sign

if __name__ == "__main__":
    input_file = ".env"
    output_file = ".env_template"
    clean_env_file(input_file, output_file)
    print(f"Cleaned .env file saved as {output_file}")