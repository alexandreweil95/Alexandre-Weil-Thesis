Those files can be used to obtain the visualization tools for any of the models.

tsne.py: projects the models' predictions for each bag on a 2D space using t-distributed stochastic neighbour embedding (t-SNE) and allows one to perform inter-subject clustering analysis.

patient_specific_clustering.py: yields the same output as tsne.py but this time for intra-patient slices clustering. Also prints the distribution of the attention weights.

show_intra_emnd.py uses t-SNE to build a clustering map showing the actual images (not just crosses or dots). For each clustering map, a corresponding plot is created where the corresponding attention weights are plotted. On the latter plot, adaptive thresholds are used to classify the attention weights into low, medium and high buckets using distinct colours.

one_plot.py: creates a single high-resolution plot of all the images in a chosen test bag, along with their learned attention weights.

vis_attention.py: individually prints all the images in a bag along with their learned attention weights. This is very similar to one_plot.py except that images are printed individually (and with the attention weight value on the image rather than above).
