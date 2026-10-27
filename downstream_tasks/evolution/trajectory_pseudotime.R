library(monocle3)
library(ggplot2)
library(umap)


###########################################################################
# please prepare the embedding and corresponding metadata before analysis #
###########################################################################

# read metadata and embedding
df_meta <- read.csv('used_meta.csv')
emb <- read.csv('emb.csv',header = F)


# set used color
mycol2 <- colorRampPalette(c("#250690","#C94479","#F5EA27"))(18)
names(mycol2) <- c("19A","19B","20A","20B","20C","21A (Delta)","21I (Delta)","21J (Delta)",
                   "21K (Omicron)","21L (Omicron)","22D (Omicron)","22F (Omicron)",
                   "23A (Omicron)","23B (Omicron)","23C (Omicron)","23D (Omicron)",
                   "23E (Omicron)","23F (Omicron)")


# construct cds object
cds <- new_cell_data_set(t(exp(emb)),cell_metadata = df_meta)
colnames(cds) <- colData(cds)$sample

# preprocessing and dimentional reduction
cds <- preprocess_cds(cds, num_dim = 100)
cds <- reduce_dimension(cds, umap.metric = 'euclidean')

# learn trajectory topology structure and pseudotime inference
cds <- cluster_cells(cds, reduction_method = 'UMAP')
cds <- learn_graph(cds)
root_cells_idx <- which(colData(cds)$clade=='19A')
cds <- order_cells(cds, root_cells=colnames(cds)[root_cells_idx[1]])

# optional: calculate spearman correlation
spearman_data <- cbind(df_meta, pseudotime(cds))
myorder <- c('19A'=1, '19B'=2, '20A'=3, '20B'= 4,
           '20C'=5, '21A (Delta)'=6,'21I (Delta)'=7, 
           '21J (Delta)'=8,'21K (Omicron)'=9,'21L (Omicron)'=10,
           '22D (Omicron)'=11,'22F (Omicron)'=12,'23A (Omicron)'=13,
           '23B (Omicron)'=14,'23C (Omicron)'= 15,'23D (Omicron)'= 16,
           '23E (Omicron)'=17,'23F (Omicron)'= 18)
spearman_data$clade_order <- myorder[spearman_data$clade]
cor(spearman_data$`pseudotime(cds)`,spearman_data$clade_order, method = 'spearman')


# plot: true label
plot_cells(cds, 
           reduction_method = 'UMAP',
           label_groups_by_cluster=FALSE,  
           color_cells_by = "clade",
           trajectory_graph_segment_size = 1.3,
           label_leaves=F,
           label_branch_points=F,
           label_roots=F,
           cell_size = 1.2,
           group_label_size = 4.5)+
  scale_color_manual(values = mycol2)+
  theme(axis.text = element_blank(),
        axis.ticks = element_blank(),
        axis.title = element_blank(),
        axis.line = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA))
ggsave('true_label.pdf',width = 6,height = 6)

# plot: pseudotime
plot_cells(cds,
           color_cells_by = "pseudotime",
           trajectory_graph_segment_size = 1.3,
           label_cell_groups=FALSE,
           label_leaves=FALSE,
           label_branch_points=FALSE,
           cell_size = 1.2)+
  guides(col='none')+
  theme(axis.text = element_blank(),
        axis.ticks = element_blank(),
        axis.title = element_blank(),
        axis.line = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA))
ggsave('pseudotime.pdf',width = 6,height = 6)
