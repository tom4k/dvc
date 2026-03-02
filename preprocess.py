import pandas as pd

def main():
    file_path = 'data/data.csv'
    # Read the CSV data
    df = pd.read_csv(file_path)
    
    # Normalize the data using Min-Max normalization
    normalized_df = (df - df.min()) / (df.max() - df.min())
    
    # Save the normalized data back
    normalized_df.to_csv(file_path, index=False)
    print(f"Successfully normalized data in {file_path}")

if __name__ == '__main__':
    main()
