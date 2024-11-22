import pickle
import gzip

# Load the existing similarity.pkl file
with open("similarity.pkl", "rb") as f:
    similarity_data = pickle.load(f)

# Compress and save as similarity.pkl.gz
with gzip.open("similarity.pkl.gz", "wb") as f:
    pickle.dump(similarity_data, f)
