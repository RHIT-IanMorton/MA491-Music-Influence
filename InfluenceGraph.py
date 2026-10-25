import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import csv
import textwrap
from SimilarityGraph import score, get_artist_vector


selected_genre = "Jazz" #Select None to make full graph

influence_graph = nx.DiGraph()

with open("data/influence_data.csv", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        
        if(selected_genre is None
           or (
              #Sets condition that influencer and follower must both have selected_genre tag
            row["influencer_main_genre"].strip() == selected_genre 
            and row["follower_main_genre"].strip() == selected_genre
              )
          ):

            influencer_id = int(row["influencer_id"])
            follower_id = int(row["follower_id"])

            influence_graph.add_node(influencer_id, 
                                     name = row["influencer_name"].strip(),
                                     genre=row["influencer_main_genre".strip()],
                                     active_start = int(row["influencer_active_start"])
                                     )
            
            influence_graph.add_node(follower_id, 
                                     name = row["follower_name"],
                                     genre=row["follower_main_genre"],
                                     active_start=int(row["follower_active_start"])
                                     )


            influencer_v = get_artist_vector(influencer_id)
            follower_v = get_artist_vector(follower_id)
            val = score(influencer_v, follower_v)
            influence_graph.add_edge(influencer_id,follower_id, weight=val)


graph_name = selected_genre if selected_genre is not None else "All genres"

print("Network:", graph_name)
print("Artists:", influence_graph.number_of_nodes())
print("Connections:", influence_graph.number_of_edges())

#Find connected groups
weak_components = list( nx.weakly_connected_components(influence_graph))

#Number of artists in largest group
largest_component_size = max(len(component) for component in weak_components)

strong_components = list(nx.strongly_connected_components(influence_graph))
largest_strong = max(len(component) for component in strong_components)

print("weakly connected components:", len(weak_components))
print("artists in largest component:", largest_component_size)
print("strongly connected components:", len(strong_components))
print("artists in largest strong:", largest_strong)


#Rank in and out degrees of each node
out_degree_rank = sorted(influence_graph.out_degree(), key=lambda item: item[1], reverse=True)

print("\nTop 10 artists by number of documented follower:")
for rank, (artist_id, degree)in enumerate(out_degree_rank[:10], start= 1):

    artist_name = influence_graph.nodes[artist_id]["name"]
    print(f"{rank}. {artist_name}: {degree}")


in_degree_rank = sorted(influence_graph.in_degree(), key=lambda item: item[1], reverse=True)

print("\nTop 10 artists by number of documented influencers:")
for rank, (artist_id, degree)in enumerate(in_degree_rank[:10], start= 1):

    artist_name = influence_graph.nodes[artist_id]["name"]
    print(f"{rank}. {artist_name}: {degree}")


display_number = 30

#Chooses top 20 artists (based on out degree) to display 
display_artists = [artist_id for artist_id, degree in out_degree_rank[:display_number]]

display_graph = influence_graph.subgraph(display_artists).copy()

for artist in nx.isolates(display_graph):
    print(
        "Isolated in displayed subset:",
        influence_graph.nodes[artist]["name"],
        "| Full-network followers:",
        influence_graph.out_degree(artist)
    )
print("\nArtists displayed:", display_graph.number_of_nodes())
print("Connections displayed:", display_graph.number_of_edges())

# Calculate positions for the selected artists.
pos = nx.spring_layout(display_graph, seed=42, k=0.8, iterations=150)
'pos = nx.circular_layout(display_graph)'

# Gets names and size nodes using out-degree in the original network.
labels = nx.get_node_attributes(display_graph, "name")

node_sizes = [
    100 + 2 * influence_graph.out_degree(artist)
    for artist in display_graph.nodes()
]

# Assign a spacing radius to each displayed node.
# Square roots are used because node_sizes represents area.
spacing_radii = {
    artist: 0.10 * np.sqrt(size / max(node_sizes))
    for artist, size in zip(display_graph.nodes(), node_sizes)
}

artists = list(display_graph.nodes())

# Repeatedly separate pairs that are too close.
for iteration in range(100):
    for i, artist_a in enumerate(artists):
        for artist_b in artists[i + 1:]:
            difference = pos[artist_b] - pos[artist_a]
            distance = np.linalg.norm(difference)

            minimum_distance = (
                spacing_radii[artist_a]
                + spacing_radii[artist_b]
                + 0.08
            )

            if distance < minimum_distance:
                if distance == 0:
                    direction = np.array([1.0, 0.0])
                else:
                    direction = difference / distance

                movement = (minimum_distance - distance) / 2

                pos[artist_a] -= movement * direction
                pos[artist_b] += movement * direction

fig, ax = plt.subplots(figsize=(16, 12))

nx.draw_networkx_nodes(
    display_graph,
    pos,
    node_size=node_sizes,
    node_color="steelblue",
    edgecolors="white", ax=ax
)

nx.draw_networkx_edges(
    display_graph,
    pos,
    node_size=node_sizes,
    arrows=True,
    arrowstyle="-|>",
    arrowsize=14,
    width=0.8,
    alpha=0.4,
    connectionstyle="arc3,rad=0.08", ax=ax
)

# Split long artist names across lines.
wrapped_labels = {
    artist: textwrap.fill(name, width=16)
    for artist, name in labels.items()
}

# Place labels slightly above their nodes.
label_positions = {
    artist: (position[0], position[1] + 0.04)
    for artist, position in pos.items()
}

nx.draw_networkx_labels(
    display_graph,
    label_positions,
    labels=wrapped_labels,
    font_size=8,
    verticalalignment="center", ax=ax,
    bbox=dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.8,
        pad=1
    )
)

edge_labels = {
    (u, v): f"{data['weight']:.2f}"
    for u, v, data in display_graph.edges(data=True)
    # if data["weight"] >= 9.5
}

nx.draw_networkx_edge_labels(
    display_graph,
    pos,
    edge_labels=edge_labels,
    font_size=6,
    font_color="darkred",
    rotate=False, ax=ax,
    bbox={"alpha": 0.7, "color": "white", "pad": 0.5}
)

plt.title(
    f"Musical Influence: {graph_name}\n"
    f"Connections among the top {display_graph.number_of_nodes()} "
    f"{graph_name} artists by out-degree"
)

ax.set_axis_off()
fig.tight_layout(pad=0.1)

ax.set_axis_off()
fig.gca().set_aspect("equal")
fig.savefig(f"data/{selected_genre}_influence.svg",  bbox_inches="tight")
plt.show()

