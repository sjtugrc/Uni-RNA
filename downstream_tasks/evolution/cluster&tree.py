import re
import os
import pickle
import torch
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio.Phylo.TreeConstruction import DistanceMatrix

###########################################################################
# please prepare the embedding and corresponding metadata before analysis #
###########################################################################

# embedding file path
emb_dir = '/home/gongyuqiao/evolution/new_16S/family_top10/used_ur_emb.pt'
mean_emb = torch.load(emb_dir)

# metadata file path
meta_dir = '/home/gongyuqiao/evolution/new_16S/family_top10/used_meta.csv'
meta = pd.read_csv(meta_dir)

# output dir
save_dir = '/home/gongyuqiao/evolution/new_16S/family_top10'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# clustering and compute ARI
data_all = pd.concat([pd.DataFrame(mean_emb.numpy()), meta], axis = 1)
emb_len = 1024
X = data_all.iloc[:,:emb_len].to_numpy()
kmeans = KMeans(n_clusters=10, random_state=2024)
kmeans.fit(X)
labels = kmeans.labels_
ARI = adjusted_rand_score(meta['family'], labels)
ARI


# tSNE dimentional reduction
tsne = TSNE(n_components=2)
X = data_all.iloc[:,:emb_len].to_numpy()
X_tsne = tsne.fit_transform(X)
data_tsne = pd.concat([pd.DataFrame(X_tsne).rename(columns={0:'t-SNE1', 1:'t-SNE2'}), meta['family']], axis = 1)


# plot
plt.figure(figsize=(10,10))
plot = sns.scatterplot(data=data_tsne, hue='family', x='t-SNE1', y='t-SNE2', edgecolor='none', alpha=0.8)
plot.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1)
plt.title(f'Uni-RNA: {round(ARI, 3)}')
plt.savefig(f'{save_dir}/uniRNA_tSNE.svg', format='svg')


# calculate distance matrix for phylo-tree construction
species_embed_cat = mean_emb.numpy()
distance_matrix = squareform(pdist(species_embed_cat, metric='cosine'))
distmat =  distance_matrix.tolist()
for i in range(len(distmat)):
    distmat[i] = distmat[i][:i+1]


# construct phylo-tree
matrix = DistanceMatrix(names=meta['id'].tolist(), matrix=distmat)
constructor = DistanceTreeConstructor()
tree = constructor.upgma(matrix)
Phylo.write(tree, f'{save_dir}/tree_ur.nwk', "newick")