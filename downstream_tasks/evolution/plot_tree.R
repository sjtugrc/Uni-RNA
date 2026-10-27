#加载包
library(ggplot2)
library(ggtree)
library(treeio)

###########################################################################
# please prepare the tree file and corresponding metadata before analysis #
###########################################################################

# read tree data and metadata
tree <- read.newick("tree_ur.nwk", node.label = "label")
meta <- read.csv('meta.csv',header = T)


# insert group information
groupInfo <- list()
groups <- names(table(meta$family))
for(i in groups){
  groupInfo[[i]]=meta$sample[which(meta$family==i)]
}
tree_ur <- groupOTU(tree,groupInfo)


#color settting
mycol <- c('Arteriviridae'='#db6968','Coronaviridae'='#4d97cd','Orthomyxoviridae'='#B78FB2',
           'Peribunyaviridae'='#459943','Phenuiviridae'='#f8984e','Tospoviridae'='#e8c559')


# plot tree
ggtree(tree_ur, branch.length = 'none', layout = "circular",
       mapping =  aes(color=group), size=1.6)+
  geom_tiplab2(size=1)+
  scale_color_manual(values = mycol)+
  guides(color='none')
ggsave(filename = 'tree_ur.svg', width = 15, height = 13)


# evaluation of the hierarchical clustering of the phylo-tree
library(mclust)
library(ape)
dist_matrix <- cophenetic.phylo(tree)
hc <- hclust(as.dist(dist_matrix))
num_clusters <- 6
clusters <- cutree(hc, k = num_clusters)
cluster_res <- data.frame(label=tree$tip.label,cluster=clusters)
meta <- merge.data.frame(meta,cluster_res,by.x = 'sample',by.y = 'label')
adjustedRandIndex(meta$family,meta$cluster)

