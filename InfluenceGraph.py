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


pos = nx.spring_layout(pop_influence,seed=42)

plt.figure(figsize=(14,10))

nx.draw_networkx_nodes(
    pop_influence,pos,node_size=10,node_color="purple"
)
nx.draw_networkx_edges(
    pop_influence,pos,arrows=True, arrowstyle="-|>",arrowsize=5,width=0.3,alpha=0.15
)

plt.title("Influence: Pop/Rock")
plt.axis("off")
plt.tight_layout()
plt.show()

