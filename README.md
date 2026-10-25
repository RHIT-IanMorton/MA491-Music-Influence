# MA491 Music Influences 2021 COMAP problem D

![Full tree](data/full_forrest.svg)

## Setup
1. Insert each problem data `.csv` file into the `data/` folder.

2. With your appropriate python environment (including `pip`), install package dependencies with
`pip install -r requirements.txt`.

If you wish to run Jupyter Notebooks like `scoring.ipynb`, instead run
`pip install -r ipynb_requirements.txt`

## Running
Simply run the command `python SimilarityGraph.py <genre> [threshold]` with your choice of genre.
If no genre is specified, the default is "Jazz". You can also pass in `full` in place of genre
to use all available data to make a complete tree. `threshold` is an optional value which determines
the threshold of similarity between two artists that must be met for the edge to stay in the graph,
the default value is 9.

WARNING - running with `full` will take a while to finish.

## Output
Within your data folder, there will be written a file with the graph data of both your
genre's similarity graph, and its maximum spanning tree.

Also available is a `.html` file which can be opened a browser of your choice displaying both the
graph and the maximum spanning tree, or a `.svg` file which is a static image of the tree.
Additionally, the file `graph_vectors.csv` contains all of the feature vectors from the latest run.

If the `full` is specified, only the maximum spanning tree data and a the `.svg` are written
since the other files would be very large or expensive to compute.
For an interactive map of the full, use another tool such as Gephi.