import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import csv

#Check of column names
'''
with open("2021_ICM_Problem_D_Data/influence_data.csv", newline = "", encoding ="utf-8-sig") as file:
    reader = csv.DictReader(file)

    print("Column names:", reader.fieldnames)
    print("First row:", next(reader))
'''
#empty graph of just pop artists
pop_influence = nx.DiGraph()

#creates nodes and edges for pop artists
with open("2021_ICM_Problem_D_Data/influence_data.csv", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        #Sets condition that influencer and follower must both have pop/rock tag
        if(row["influencer_main_genre"].strip() =="Pop/Rock" and row["follower_main_genre"].strip() =="Pop/Rock"):

            influencer_id = int(row["influencer_id"])
            follower_id = int(row["follower_id"])

            pop_influence.add_node(influencer_id, name = row["influencer_name"])
            pop_influence.add_node(follower_id, name = row["follower_name"])

            pop_influence.add_edge(influencer_id,follower_id)



#Draws a graph of all pop connections (Very Laggy)
'''
pos = nx.spring_layout(pop_influence,seed=42)

plt.figure(figsize=(14,10))

nx.draw_networkx_nodes(
    pop_influence,pos,node_size=10,node_color="purple"
)
nx.draw_networkx_edges(
    pop_influence,pos,arrows=True, arrowstyle="-|>",arrowsize=5,width=0.3,alpha=0.15
)
'''

#Makes graph of the top 30 pop artists
top_artists = sorted(pop_influence.nodes(), key=lambda artist: pop_influence.out_degree(artist), reverse = True)[:50]

top_pop_influence = pop_influence.subgraph(top_artists).copy()

pos = nx.spring_layout(top_pop_influence,seed=42,k=1.5,iterations=100)
labels=nx.get_node_attributes(top_pop_influence,"name")
plt.figure(figsize=(16,12))

node_sizes = [100+2*pop_influence.out_degree(artist) for artist in top_pop_influence.nodes()]

nx.draw_networkx_nodes(
    top_pop_influence,pos,node_size=node_sizes,node_color="purple"
)
nx.draw_networkx_edges(
    top_pop_influence,pos,arrows=True, arrowstyle="-|>",arrowsize=10,width=0.6,alpha=0.3
)
nx.draw_networkx_labels(
    top_pop_influence,pos,labels=labels,font_size=8
)

plt.title("Top Influence: Pop/Rock")
plt.axis("off")
plt.tight_layout()
plt.show()

