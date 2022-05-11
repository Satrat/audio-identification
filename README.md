# Audio Identification with Spectral Peaks

An audio identification system for classical and pop music using spectral peaks as a feature set. Audio fingerprints of each database recording are stored as a pariwise inverted index has table for efficiency. The system returns the top 3 database matches for each query, and achieves an average precision of 81.4% on the classical and rock genres of the GTZAN dataset. The average computation time is 5.25 seconds per query.

The GTZAN dataset can be downloaded from: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification



## Directory Layout
### music_id.py
Audio identification system implementation

### example_usage.ipynb
demonstration of running the audio ID algorithm on a folder queries

### experiments_graphs.ipynb
Plots output at different stages/variations of the algorithm and reports overall precision on GTZAN
