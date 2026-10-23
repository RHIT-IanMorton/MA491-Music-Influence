"""
genre_graph.py - Ian Morton, AI assisted

Scraps through influence_data.csv to find all unique artist_ids of a given genre.

Creates a feature vector for each artist_id with data from data_by_artist.csv.

Creates a graph with all artists being nodes, and edges between each artist being
weighted by the similarity as calculated with score() between their feature vectors, only including edges
with a weight of at least SIMILARITY_THRESHOLD which ranges higher in 0 to 1.


"""




import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import numpy as np

# =========================
# Configuration
# =========================

DATA_FOLDER = "data/"
INFLUENCE_FILE = DATA_FOLDER + "influence_data.csv"
ARTIST_FILE = DATA_FOLDER + "data_by_artist.csv"
FEATURE_FILE = DATA_FOLDER + "artist_vectors.csv"

GENRE = "Jazz"
SIMILARITY_THRESHOLD = 3

# Choose the columns from data_by_artist.csv that form each vector.
# Replace these examples with columns that actually exist in your file.
FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "loudness",
    "mode",
    "key",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "duration_ms",
    ]

OUTPUT_GRAPH = DATA_FOLDER + "artist_similarity.graphml"
OUTPUT_FEATURES = DATA_FOLDER + "graph_vectors.csv"

def make_feature_file():
    artist_ids = get_all_ids(ARTIST_FILE)
    print(f"Found {len(artist_ids)} unique IDs.")

    vectors = get_artist_vectors(
        artist_ids,
        ARTIST_FILE,
        FEATURE_COLUMNS,
        standardize=True
    )

    print(f"Features used: {list(vectors.columns)}")

    # Save the vectors used to build the graph.
    vectors.to_csv(FEATURE_FILE)


# =========================
# 1. Extract artist IDs
# =========================

def get_genre_ids(influence_file, genre):
    df = pd.read_csv(influence_file)

    influencer_ids = df.loc[
        df["influencer_main_genre"].eq(genre),
        "influencer_id"
    ]

    follower_ids = df.loc[
        df["follower_main_genre"].eq(genre),
        "follower_id"
    ]

    # Combine IDs from both roles and remove duplicates.
    ids = pd.concat(
        [influencer_ids, follower_ids],
        ignore_index=True
    ).dropna()

    # Normalize IDs so values like 123 and "123" match.
    return set(ids.astype(str).str.strip()) - {""}

def get_all_ids(artist_file):
    df = pd.read_csv(artist_file)
    id = df["artist_id"].dropna()
    return set(id.astype(str).str.strip()) - {""}


# =========================
# 2. Build artist vectors
# =========================


# def get_artist_vectors(artist_file, artist_ids, features):
#     df = pd.read_csv(artist_file)

#     required_columns = ["artist_id"] + features
#     missing_columns = set(required_columns) - set(df.columns)

#     if missing_columns:
#         raise ValueError(
#             f"Columns not found in {artist_file}: "
#             f"{sorted(missing_columns)}"
#         )

#     df["artist_id"] = df["artist_id"].astype("string").str.strip()

#     # Keep only artists in the requested genre.
#     df = df[df["artist_id"].isin(artist_ids)].copy()

#     # Ensure each artist has only one feature vector.
#     # Stop if artist_id is not unique rather than choosing a row arbitrarily.
#     if df["artist_id"].duplicated().any():
#         duplicates = df.loc[
#             df["artist_id"].duplicated(keep=False), "artist_id"
#         ].unique()

#         raise ValueError(
#             f"Duplicate artist_id values found: {list(duplicates[:10])}. "
#             "Resolve duplicate rows before building the graph."
#         )

#     # Use artist IDs as the index.
#     df = df.set_index("artist_id")

#     # Extract the selected features.
#     vectors = df[features].copy()

#     # Convert selected features to numeric values.
#     for column in features:
#         vectors[column] = pd.to_numeric(
#             vectors[column], errors="coerce"
#         )

#     # Replace infinite values with missing values.
#     vectors = vectors.replace([float("inf"), float("-inf")], float("nan"))

#     # Drop features that contain no usable numeric values.
#     vectors = vectors.dropna(axis=1, how="all")

#     if vectors.shape[1] == 0:
#         raise ValueError("No usable numeric feature columns remain.")

#     # Fill missing values with each feature's median.
#     vectors = vectors.fillna(vectors.median())

#     # Cosine similarity is undefined for zero vectors.
#     zero_vectors = vectors.eq(0).all(axis=1)
#     if zero_vectors.any():
#         zero_ids = vectors.index[zero_vectors].tolist()
#         print(
#             f"Warning: excluding {len(zero_ids)} artists "
#             "whose vectors contain only zeros."
#         )
#         vectors = vectors.loc[~zero_vectors]

#     return vectors

def get_artist_vectors(artist_ids, artist_file=FEATURE_FILE, features=FEATURE_COLUMNS, standardize=False):
    df = pd.read_csv(artist_file, dtype={"artist_id": str})

    df["artist_id"] = df["artist_id"].str.strip()
    artist_ids = {str(i).strip() for i in artist_ids}

    # Keep only requested artists
    df = df[df["artist_id"].isin(artist_ids)].copy()

    if df["artist_id"].duplicated().any():
        raise ValueError("Duplicate artist_id values found.")

    # Extract numeric features
    vectors = df.set_index("artist_id")[features].apply(
        pd.to_numeric, errors="coerce"
    )

    vectors = vectors.replace([np.inf, -np.inf], np.nan)
    vectors = vectors.dropna(axis=1, how="all")
    vectors = vectors.fillna(vectors.median())

    # Remove artists with no usable feature values
    vectors = vectors.dropna(axis=0, how="any")

    if standardize:
    # Standardize features across artists (mean=0, std=1)
        scaler = StandardScaler()
        standardized = scaler.fit_transform(vectors)

        return pd.DataFrame(
            standardized,
            index=vectors.index,
            columns=vectors.columns
        )

    return vectors

def get_artist_vector(artist_id, artist_file=FEATURE_FILE, features=FEATURE_COLUMNS):
    """Return the feature vector for a given artist_id."""

    artist_id = str(artist_id)
    df = pd.read_csv(artist_file, dtype={"artist_id": str})

    # Find the artist's row
    artist = df.loc[df["artist_id"] == str(artist_id)]

    if artist.empty:
        raise ValueError(f"Artist ID {artist_id} not found.")

    if len(artist) > 1:
        raise ValueError(f"Multiple rows found for artist ID {artist_id}.")

    # Extract the selected fields as a numeric vector
    vector = artist.iloc[0][features]
    vector = pd.to_numeric(vector, errors="raise")

    return vector.to_numpy(dtype=float)

def score(v1, v2, method="euclidean", sigma=1.0):
    v1 = np.asarray(v1, dtype=float).ravel()
    v2 = np.asarray(v2, dtype=float).ravel()


    if v1.shape != v2.shape:
        raise ValueError("Vectors must have the same shape.")

    if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
        raise ValueError("Vectors must contain only finite values.")

    distance = np.linalg.norm(v1 - v2)

    if method == "euclidean":
        return float(10/(1+distance))

    elif method == "gaussian":
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        return float(np.exp(-(distance ** 2) / (2 * sigma ** 2)))

    else:
        raise ValueError(
            "method must be 'euclidean' or 'gaussian'."
        )

def artist_similarity(artist_id_1, artist_id_2):
    """Return similarity between two artists."""

    vector_1 = get_artist_vector(artist_id_1)
    vector_2 = get_artist_vector(artist_id_2)

    return float(
        score(
            vector_1,
            vector_2
        )
    )

# =========================
# 3. Build the graph
# =========================

def build_similarity_graph(vectors):
    graph = nx.Graph()

    artist_ids = vectors.index.tolist()
    vectors = vectors.to_numpy()

    # Add one node for each artist.
    for artist_id in artist_ids:
        graph.add_node(artist_id)


    # Add an undirected weighted edge between every pair.
    for i, j in combinations(range(len(artist_ids)), 2):
        similarity = score(vectors[i], vectors[j])
        if (similarity >= SIMILARITY_THRESHOLD):     # drop edges too small
            graph.add_edge(
                artist_ids[i],
                artist_ids[j],
                weight=float(similarity)
            )

    return graph


# =========================
# 4. Run everything
# =========================

def main(genre):
    make_feature_file()
    artist_ids = get_genre_ids(INFLUENCE_FILE, genre)
    print(f"Found {len(artist_ids)} unique {genre} IDs.")

    vectors = get_artist_vectors(
        artist_ids,
        FEATURE_FILE,
        FEATURE_COLUMNS
    )

    print(f"Matched {len(vectors)} artists in {ARTIST_FILE}.")
    print(f"Features used: {list(vectors.columns)}")

    # Save the vectors used to build the graph.
    vectors.to_csv(OUTPUT_FEATURES)

    graph = build_similarity_graph(vectors)

    # Save the graph in GraphML format.
    nx.write_graphml(graph, OUTPUT_GRAPH)

    print(f"Graph nodes: {graph.number_of_nodes()}")
    print(f"Graph edges: {graph.number_of_edges()}")
    print(f"Saved graph to {OUTPUT_GRAPH}")
    print(f"Saved vectors to {OUTPUT_FEATURES}")

    # Display a few example edges.
    print("\nExample artist similarities:")
    for u, v, data in list(graph.edges(data=True))[:10]:
        print(f"{u} -- {v}: {data['weight']:.4f}")


if __name__ == "__main__":
    main(GENRE)

