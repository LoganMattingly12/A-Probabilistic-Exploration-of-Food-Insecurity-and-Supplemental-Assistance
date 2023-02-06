library(dendextend)
(library(dplyr))
(library(ggplot2))

#Data Extraction
set.seed(569)
df <- as.data.frame(read.csv('C:\\Users\\ltmat\\Documents\\Logan\\School\\MSU\\Thesis\\Data\\ETL Data\\Demographics 1.csv'))
print(df)
any(is.na(df))
summary(df)
count(df)

#Data Cleaning and Standarizing
labels1 <- df$Household_Number
df$Household_Number <- NULL
df <- as.data.frame(scale(df))
df <- subset(df, select = -c(SNAP))
#df <- df[df$Food_Sufficient==3,]
summary(df)
str(df)
print(df)
count(df)


#First CLustering
dist_matrix <- dist(df, method = 'euclidean')
hclust_1 <- hclust(dist_matrix, method = 'complete')
plot(hclust_1)


#Cluster recolor
plot(hclust_1,labels = labels1)
rect.hclust(hclust_1, k=8, border=2:6)
avg_dend_obj <- as.dendrogram(hclust_1)
avg_col_dend <- color_branches(avg_dend_obj, k =8 )
plot(avg_col_dend)

#Cluster Maping
cut_avg <- cutree(hclust_1, k = 8)
df_cl <- mutate(df, cluster = cut_avg)
count(df_cl,cluster)
csv = table(df_cl$cluster,labels)
csv = as.data.frame(csv)
csv = csv[csv$Freq==1,]
csv <- subset(csv, select = -c(Freq))
print(csv)
write.csv(csv,'C:\\Users\\ltmat\\Documents\\Logan\\School\\MSU\\Thesis\\Data\\ETL Data\\Demographics_Clusters.csv')
#Exploratory Analysis
ggplot(df_cl, aes(x=Income, y = Food_Sufficient, color = factor(cluster))) + geom_point()



