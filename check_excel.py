import pandas as pd

# Read the Excel file
filename = 'spotify_metadata_Pop_Hits_2025__Top_50_.xlsx'
print(f"Reading file: {filename}")

try:
    df = pd.read_excel(filename)
    
    # Print basic information
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Number of tracks: {len(df)}")
    
    # Print column names
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")
    
    # Print the first few tracks
    print("\nFirst 5 tracks:")
    for i, row in df.head().iterrows():
        print(f"{i+1}. {row['track_name']} by {row['artist']} (Popularity: {row['popularity']})")
    
except Exception as e:
    print(f"Error reading Excel file: {e}") 