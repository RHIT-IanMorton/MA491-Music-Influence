"""
genre_graph.py - Ian Morton, AI assisted

Scraps through influence_data.csv to find all unique artist_ids of a given genre.

Creates a feature vector for each artist_id with data from data_by_artist.csv.

Creates a graph with all artists being nodes, and edges between each artist being
weighted by the similarity as calculated with score() between their feature vectors, only including edges
with a weight of at least SIMILARITY_THRESHOLD which ranges higher in 0 to 1.


"""


from sys import argv
from pathlib import Path
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from pyvis.network import Network

# =========================
# Configuration
# =========================

DATA_FOLDER = "data/"
INFLUENCE_FILE = DATA_FOLDER + "influence_data.csv"
ARTIST_FILE = DATA_FOLDER + "data_by_artist.csv"
FEATURE_FILE = DATA_FOLDER + "artist_vectors.csv"

GENRE = "Jazz"
SIMILARITY_THRESHOLD = 9

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
    # "popularity"
    ]

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

def get_artist_vectors(artist_ids, feature_file=FEATURE_FILE, features=FEATURE_COLUMNS, standardize=False) -> np.ndarray:
    df = pd.read_csv(feature_file, dtype={"artist_id": str})

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

def get_artist_vector(artist_id, feature_file=FEATURE_FILE, features=FEATURE_COLUMNS) -> np.ndarray:
    """Return the feature vector for a given artist_id."""

    artist_id = str(artist_id)
    df = pd.read_csv(feature_file, dtype={"artist_id": str})

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

# ==============
#    SCORING
# ==============

def score(v1, v2, method="cosine", sigma=1.0):
    v1 = np.asarray(v1, dtype=float).ravel()
    v2 = np.asarray(v2, dtype=float).ravel()


    if v1.shape != v2.shape:
        raise ValueError("Vectors must have the same shape.")

    if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
        raise ValueError("Vectors must contain only finite values.")

    if method == "cosine":
        return 10 * np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    if method == "euclidean":
        distance = np.linalg.norm(v1 - v2)
        return float(100/(10+distance))

    if method == "minkowski":
        distance = np.linalg.norm(v1 - v2, 1)
        return float(100/(10 + distance))

    elif method == "gaussian":
        distance = np.linalg.norm(v1 - v2)
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

    # remove isolated nodes
    isolated_nodes = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated_nodes)

    return graph

def draw_graph(graph: nx.Graph, file, show=False):
    """
    Draws the graph with matplotlib to be outputed to file named with genre

    NOTE - save as `.svg` 
    """

    # Consistent layout; stronger weights attract nodes more strongly
    pos = nx.nx_agraph.graphviz_layout(
        graph,
        prog="sfdp",
        args="-Gpack=true -Gpackmode=cluster -GK=0.8 -Gsep=-5"
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scale edge widths to similarity scores
    weights = [graph[u][v]["weight"] for u, v in graph.edges()]
    w_min, w_max = min(weights, default=0), max(weights, default=1)

    widths = [
        0.4 + 1.5 * (w - w_min) / (w_max - w_min)
        if w_max > w_min else 1.0
        for w in weights
    ]

    nx.draw_networkx_edges(
        graph, pos, ax=ax,
        edge_color="gray",
        width=widths,
        alpha=0.45
    )

    nx.draw_networkx_nodes(
        graph, pos, ax=ax,
        node_size=[
            3 + 2 * graph.degree(n) for n in graph.nodes()
        ],
        node_color="#01B8E6",
        alpha=0.85,
        linewidths=0
    )

    # Label only the 15 highest-degree artists to avoid clutter
    top_nodes = sorted(
        graph.degree, key=lambda item: item[1], reverse=True
    )[:15]
    labels = {node: node for node, _ in top_nodes}

    nx.draw_networkx_labels(
        graph, pos,
        labels=labels,
        font_size=7,
        ax=ax
    )

    if not graph.number_of_edges() > 1000:
        edge_labels = {
            (u, v): f"{data['weight']:.2f}"
            for u, v, data in graph.edges(data=True)
            if data["weight"] >= 9.5
        }

        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=edge_labels,
            font_size=6,
            font_color="darkred",
            rotate=False,
            bbox={"alpha": 0.7, "color": "white", "pad": 0.5}
        )

    ax.set_axis_off()
    fig.tight_layout(pad=0.1)

    # fig.savefig("artist_network.pdf", bbox_inches="tight")
    # fig.savefig("artist_network.png", dpi=300, bbox_inches="tight")
    fig.savefig(file, bbox_inches="tight")

    if show:
        plt.show()

def check_max_weight_influence(G, influence_file=INFLUENCE_FILE):
    influence_df = pd.read_csv(influence_file)

    # Store each influence pair for fast lookup, ignoring direction
    influence_pairs = {
        frozenset((row["influencer_name"], row["follower_name"]))
        for _, row in influence_df.iterrows()
    }

    results = []

    for node in G.nodes():
        edges = G.edges(node, data=True)

        if not edges:
            continue

        # Find this node's highest-weight edge
        u, v, data = max(edges, key=lambda edge: edge[2].get("weight", float("-inf")))

        pair = frozenset((u, v))
        results.append({
            "node": node,
            "other_node": v if u == node else u,
            "weight": data.get("weight"),
            "artist_pair": (u, v),
            "influence_exists": pair in influence_pairs,
        })

    return pd.DataFrame(results)

def check_all_edges_influence(G, influence_file=INFLUENCE_FILE):
    influence_df = pd.read_csv(influence_file)

    # Build a set of artist pairs, ignoring direction
    influence_pairs = {
        frozenset((row["influencer_name"], row["follower_name"]))
        for _, row in influence_df.iterrows()
    }

    results = []

    for u, v, data in G.edges(data=True):
        pair = frozenset((u, v))

        results.append({
            "artist_1": u,
            "artist_2": v,
            "weight": data.get("weight"),
            "influence_exists": pair in influence_pairs,
        })

    return pd.DataFrame(results)

# =========================
# 4. Run everything
# =========================

def main(genre: str):
    if not Path(FEATURE_FILE).exists():
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

    # ========== RELABELLING ================

    # Load artist names
    artist_names = (
        pd.read_csv(ARTIST_FILE, dtype={"artist_id": str})
        .set_index("artist_id")["artist_name"]
        .to_dict()
    )

    # Relabel graph nodes (assuming node IDs are artist IDs)
    graph = nx.relabel_nodes(
        graph,
        {node: artist_names.get(str(node), str(node)) for node in graph.nodes()}
    )

    # ========== RELABELLING ================

    # Save the graph in GraphML format.
    genre = genre.replace("/", "")
    nx.write_graphml(graph, f"{DATA_FOLDER}{genre}_graph.graphml")
    # draw_graph(graph, f"{DATA_FOLDER}{genre}_graph.svg")


    print(f"Graph nodes: {graph.number_of_nodes()}")
    print(f"Graph edges: {graph.number_of_edges()}")
    print(f"Saved graph to {DATA_FOLDER}{genre}_graph.graphml")
    print(f"Saved vectors to {OUTPUT_FEATURES}")

    # Display a few example edges.
    print("\nExample artist similarities:")
    for u, v, data in list(graph.edges(data=True))[:10]:
        print(f"{u} -- {v}: {data['weight']:.4f}")

    forrest = nx.maximum_spanning_tree(graph, weight="weight")
    nx.write_graphml(forrest, f"{DATA_FOLDER}{genre}_forrest.graphml")
    forrest = forrest.subgraph(
        node for component in nx.connected_components(forrest)
        if len(component) > 5
        for node in component
    ).copy()
    draw_graph(forrest, f"{DATA_FOLDER}{genre}_forrest.svg")


    print("\n")
    print("Original nodes:", graph.number_of_nodes())
    print("Original edges:", graph.number_of_edges())
    print("Forest nodes:", forrest.number_of_nodes())
    print("Forest edges:", forrest.number_of_edges())
    print("Connected components:", nx.number_connected_components(forrest))
    print(f"Saved forrest to {DATA_FOLDER}{genre}_forrest.graphml")


    net = Network(notebook=True, height="100vh", width="100vw", cdn_resources='in_line')
    net.from_nx(graph)
    
    for edge in net.edges:
        w = edge["width"]

        edge["label"] = f"{w:.3f}"
        edge["title"] = f"Similarity: {w:.4f}"

        # Higher similarity -> shorter edge
        edge["length"] = 50 + (10 - w) * 100/(10-SIMILARITY_THRESHOLD)

    # Force-directed layout handles physics calculations efficiently
    net.show(f"{DATA_FOLDER}{genre}_graph.html")

    net = Network(notebook=True, height="100vh", width="100vw", cdn_resources='in_line')
    net.from_nx(forrest)

    for edge in net.edges:
        w = edge["width"]

        edge["label"] = f"{w:.3f}"
        edge["title"] = f"Similarity: {w:.4f}"

        # Higher similarity -> shorter edge
        edge["length"] = 50 + (10 - w) * 100/(10-SIMILARITY_THRESHOLD)

    # Force-directed layout handles physics calculations efficiently
    net.show(f"{DATA_FOLDER}{genre}_forrest.html")

def main_full():
    if not Path(FEATURE_FILE).exists():
            make_feature_file()
    artist_ids = get_all_ids(ARTIST_FILE)
    print(f"Found {len(artist_ids)} unique IDs.")

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

    # ========== RELABELLING ================

    # Load artist names
    artist_names = (
        pd.read_csv(ARTIST_FILE, dtype={"artist_id": str})
        .set_index("artist_id")["artist_name"]
        .to_dict()
    )

    # Relabel graph nodes (assuming node IDs are artist IDs)
    graph = nx.relabel_nodes(
        graph,
        {node: artist_names.get(str(node), str(node)) for node in graph.nodes()}
    )

    # ========== RELABELLING ================

    # Save the graph in GraphML format.
    nx.write_graphml(graph, f"{DATA_FOLDER}full_graph.graphml")
    # draw_graph(graph, f"{DATA_FOLDER}{genre}_graph.svg")


    print(f"Graph nodes: {graph.number_of_nodes()}")
    print(f"Graph edges: {graph.number_of_edges()}")
    # print(f"Saved graph to {DATA_FOLDER}full_graph.graphml")
    print(f"Saved vectors to {OUTPUT_FEATURES}")

    # Display a few example edges.
    print("\nExample artist similarities:")
    for u, v, data in list(graph.edges(data=True))[:10]:
        print(f"{u} -- {v}: {data['weight']:.4f}")

    forrest = nx.maximum_spanning_tree(graph, weight="weight")
    nx.write_graphml(forrest, f"{DATA_FOLDER}full_forrest.graphml")
    forrest = forrest.subgraph(
        node for component in nx.connected_components(forrest)
        if len(component) > 5
        for node in component
    ).copy()
    draw_graph(forrest, f"{DATA_FOLDER}full_forrest.svg")


    print("\n")
    print("Original nodes:", graph.number_of_nodes())
    print("Original edges:", graph.number_of_edges())
    print("Forest nodes:", forrest.number_of_nodes())
    print("Forest edges:", forrest.number_of_edges())
    print("Connected components:", nx.number_connected_components(forrest))
    print(f"Saved forrest to {DATA_FOLDER}full_forrest.graphml")

if __name__ == "__main__":
    if len(argv) < 2:
        raise BufferError("Please specify either 'full', or a genre from influence_data.csv")

    if argv[1] == "full":
        main_full()
    else:
        main(str(argv[1]))

