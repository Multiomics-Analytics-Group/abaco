# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # `ABaCo` demo: No biological labels w/ Mgnify tomatoes

# %% [markdown]
# In this tutorial we demonstrate how ABaCo can be used: 
# - without known biological group labels (e.g., only correct batch effect)
# - and for integrating samples from multiple studies.
#
# Specifically, we integrate tomato rhizosphere metagenomics analyses available on MGniFy. 
# - MGYS00006231, Illumina HiSeq 4000
# - MGYS00006204, Illumina HiSeq 2500
# - MGYS00006205, Illumina HiSeq 2500
#
# **Additionally** we provide an intro to [working with anndata](https://anndata.readthedocs.io/en/stable/tutorials/notebooks/getting-started.html)  
#
# **Note** in this tutorial we include a dummy bio column for demo purposes, but it is not necessary to have a bio column to use abaco. 
#
# -----
# **Goal:**
#
# With ABaCo the aim is to remove the technical variation between sequencing platforms. 
#
# -----
#
# To start we retrieve the data from MGniFy. 

# %% [markdown]
# ## Downloading the data

# %%
import requests 
from io import StringIO
import pandas as pd
import numpy as np

# urls manually retrieved from mgnify website
datasets = {
    "MGYS00006205": "https://www.ebi.ac.uk/metagenomics/api/v1/studies/MGYS00006205/pipelines/5.0/file/ERP140107_taxonomy_abundances_SSU_v5.0.tsv",
    "MGYS00006231": "https://www.ebi.ac.uk/metagenomics/api/v1/studies/MGYS00006231/pipelines/5.0/file/ERP139927_taxonomy_abundances_SSU_v5.0.tsv",
    "MGYS00006204": "https://www.ebi.ac.uk/metagenomics/api/v1/studies/MGYS00006204/pipelines/5.0/file/ERP140102_taxonomy_abundances_SSU_v5.0.tsv",
}
# init for dfs
dfs = {}
for key, url in datasets.items():
    # download
    response = requests.get(url)
    # treat as file
    file_like = StringIO(response.content.decode('utf-8'))
    # read into pd df
    dfs[key] = pd.read_csv(file_like, sep='\t')
    # rename sampleid col to taxa
    dfs[key] = dfs[key].rename(columns={"#SampleID": "taxa"})
    # set index to taxa
    dfs[key] = dfs[key].set_index('taxa')
    # drop rows with all zeros
    dfs[key] = dfs[key][dfs[key].sum(axis=1)>0]
    # sanity check
    print(dfs[key].shape)
    display(dfs[key].sample(5))

# %% [markdown]
# Based on the sampled outputs printed above, w can see that the data: 
# - has taxa as rows
# - the samples as columns 
# - and the values are the number of taxanomic assignments for the given taxa x sample
#
# To join them we will use a helper function `abaco.utils.df_joiner()`. 

# %%
from abaco.utils import df_joiner

joined = df_joiner(
    df_dict=dfs, 
    on="taxa",
    how="outer",
)

joined.info()

# %% [markdown]
# ## A brief intro to `anndata`
#
# ![img](https://raw.githubusercontent.com/scverse/anndata/main/docs/_static/img/anndata_schema.svg)
#
# For this example we will use anndata to store multiple layers of the data (e.g., raw counts, normalized, etc.) and annotate the samples and taxa with metadata. Below we prepare the annotations (i.e., obs, var).  

# %%
import anndata as ad
import scanpy as sc

# preparing obs metadata
dfs_T = {}
# for each study
for key in dfs:
    # transpose
    dfs_T[key] = dfs[key].T
    # add study source column to be batch effect
    dfs_T[key]['source'] = key
    # dummy bio column, rando group assign, for checking and demo 
    dfs_T[key]['bio'] = np.random.choice(["Group1", "Group2"], size = len(dfs_T[key]))
    # drop allother cols
    dfs_T[key] = dfs_T[key][['source', 'bio']]
# now concat all study metadata 
obs = pd.concat(dfs_T.values())
print(obs.shape)
display(obs.sample(5))

# preparing var metadata (taxa levels)
# get taxa index as df and split on ';'
var = joined.reset_index()[['taxa']]
var = var['taxa'].str.split(';', expand=True)
# rename cols 
var.columns = [
    "superkingdom", "kingdom","phylum", "class", 
    "order", "family", "genus", "species"
]
print(var.shape)
display(var.sample(5))

# %% [markdown]
# We can initiate the AnnData objct with the values (taxanomic assignment counts) and the metadata we prepped above. 

# %%
# init anndata object
full = ad.AnnData(
    X=joined.T.values,
    obs=obs,
    var=var,    
)
# check it out
full

# %% [markdown]
# With anndata we can also store additional layers seamlessly with the same annoations. Below we will create a layer with the NaNs filled with 0s

# %%
# layer with no na
full.layers['nona'] = np.nan_to_num(full.X, nan=0)
print(full)

# checking it out, also demoing to_df method
print("original:")
display(full.to_df().head(5))
print("no na:")
display(full.to_df(layer='nona').head(5))

# %% [markdown]
# we can also add normalized and clr layers. For more details on the reasoning of these layers visit our other tutorials.

# %%
from skbio.stats.composition import clr 

full.layers['norm'] = full.layers['nona']/full.layers['nona'].sum(axis=1).reshape(-1,1)

# CLR transform normalized data, replacing zeros with smallll val to avoid log(0)
full.layers['clr'] = clr(np.where(full.layers['norm'] > 0, full.layers['norm'], 1e-10))

print(full)

# %% [markdown]
# With anndata we can also filter the data using the annotations/metadata. Below we will keep the variables (taxa) with genus assignment. Additionally we filter out some taxa if they are missing in more than 30% of the samples. 

# %%
# must have genus level
adata = full[:, ~full.var['genus'].isna()]
print(adata)

# filtering if taxa is missing (NaN) in more than 30% of samples
adata = adata[:, np.isnan(adata.X).sum(axis=0)/adata.X.shape[0] <= 0.3]
print(adata)

# %% [markdown]
# We are down to 176 taxa. 
#
# However, it can be trickier to aggregate with anndata -- but we can use the [scanpy toolkit](https://scanpy.readthedocs.io/en/stable/index.html) to help.  
#
# Specifically we will use [scanpy.get.aggregate()](https://scanpy.readthedocs.io/en/stable/generated/scanpy.get.aggregate.html) to help us check that the dummy biological groups we created are not different. 

# %%
# agg by fake bio group 
agg_mean = sc.get.aggregate(
    # the anndata object
    adata=adata,
    # what layer to use
    layer='clr',
    # group by col
    by="bio",
    # axis that the group lives
    axis="obs",
    # how to agg
    func='mean',
)
# check out the returned AnnData objct
print(agg_mean)
# we can see taht the default layer name if not provided is the func name 
display(agg_mean.to_df('mean').head(5))

# %% [markdown]
# Now lets quickly check that group 1 and 2 are not different visually and quick stats.

# %%
import plotly.graph_objects as go
from scipy import stats

# transpose for easier plotting
bio_sum = agg_mean.to_df('mean').T

# init fig
fig = go.Figure()
# add traces
fig.add_trace(go.Histogram(x=bio_sum['Group1'], name='Group1'))
fig.add_trace(go.Histogram(x=bio_sum['Group2'], name='Group2'))

# formatting
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.show()

# check no diff in stats 
stats.wilcoxon(bio_sum['Group1'], bio_sum['Group2'])

# %% [markdown]
# ## Prepare data for ABaCo
#
# Great. Now lets proceed with adata where we can easily extract a df to meet the data format required for abaco.

# %%
batch_col = 'source'
bio_col = 'bio'
id_col = 'index'

# the raw counts as df
df_taxa = adata.to_df("nona")

# appending on the categorical data of interest
df_all = pd.concat([df_taxa, adata.obs[[batch_col, bio_col]]], axis=1).reset_index()

df_all['bio'] = df_all['bio'].astype('category')
df_all['source'] = df_all['source'].astype('category')

df_all.info()

# %% [markdown]
# **`pd.DataFrame` Requirements for ABaCo:**
#
# The dataset contains the following making it compatible with the ABaCo framework:
#
# | id_col | batch_col  | bio_col  | count1 | count2 | ... |
# |--------|------------|----------|--------|--------|-----|
# | A      | 24/07/2025 | RA       | #      | #      | ... |
# | B      | 15/06/2024 | RD       | #      | #      | ... |
# | C      | 24/07/2025 | RL       | #      | #      | ... |
#
# - The data has categorical columns: 
#     1. unique ids to identify the observations/samples e.g. sample id col
#     2. ids for the batch/factor groupings to be corrected by abaco. e.g. our phony bio data 
#     3. biological/experimental factor variation for abaco to retain when correcting batch effect e.g., study id 
#
# - And taxa counts to be trained on. 
#
# We can use `abaco.plots.plotPCoA()` to visualize any batch and biological effects based on the given categories. 

# %%
from abaco.plots import plotPCoA
import plotly.io as pio
pio.renderers.default = "notebook"

plotPCoA(
    data=df_all, 
    sample_label=id_col, 
    batch_label=batch_col,  
    experiment_label=bio_col,
)

# %% [markdown]
# - Batch effect (colours): 
#     - Different sequencing platforms which could result in a technical source of variation captured by the clustering of studies 6204 & 6205 (Illumina HiSeq 2500) vs. study 6231 (Illumina HiSeq 4000) along PCo1
#
# - Biological effect (shapes): 
#     - The groups were randomly assigned so we expect no clustering by shape as supported by the pcoa above. This behaviour should not change with abaco reconstruction.
# -----

# %% [markdown]
# ## The goal 
#
# Here the aim of **ABaCo** is to: 
# 1) correct the batch effect (e.g., the points should no longer cluster by colour in the PCoA) while
# 2) maintaining biological variance (or lack thereof).
#
# Ideally, after using AbaCo to transform the data, the resulting PCoA coloured by batches will look like a colourful mixture of points.
#
# -----
#
# ## Using `ABaCo`
#
# ### Setting up ABaCo
#
# We instantiate the `abaco.metaABaCo()` class and pass the required parameters shown in the cell below. 
#
# Usually, setup of the parameters is required, which are explained in brief in the documentation e.g. `help(metaABaCo)`
#
#

# %%
from abaco.ABaCo import metaABaCo
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instaantiate the ABaCo model
model = metaABaCo(
    data=df_all, # Pre-processed dataframe
    n_bios=df_all[bio_col].nunique(), # Number of biological groups in the data
    bio_label=bio_col, # Column where biological groups are labeled in the dataframe
    n_batches=df_all[batch_col].nunique(), # Number of batch groups in the data
    batch_label=batch_col, # Column where batch groups are labeled in the dataframe
    n_features=df_taxa.shape[1], # Number of features (taxonomic groups)
    prior="MoG", # Prior distribution 
    device=device, # Device
    d_z=16, # default dim of latent space
    epochs = [3000, 1000, 3000], # num epochs for each training phase
    disc_net=[256, 128, 64], # stronger discriminator
    disc_act_fun=torch.nn.LeakyReLU(0.1),
)

# %% tags=["hide-output"]
help(metaABaCo)

# %% [markdown]
# ### Training the ABaCo model
#
# To train ABaCo on the prepared dataset, we then use method `abaco.metaABaCo.fit()`. 

# %% tags=["hide-output"]
# Train the model,
model.fit(
  seed=42,
  w_elbo_nll=10, # more emphasis on reconstruction
  w_bio_penalty=0.0, # disable bio supervision
  w_cluster_penalty=0.0, # light regularization or can be 0
  w_disc=1.0, # default, discriminator training strength
)


# %% [markdown]
# `abaco.metaABaCo` provides a method that visualizes the latent vectors via PCA for dimensionality reduction: `.plot_pca_posterior()`

# %%
model.plot_pca_posterior()

# %% [markdown]
# ### Reconstructing the dataset with ABaCo
#
# To reconstruct the dataset we use method `abaco.metaABaCo.correct()` and save it as a new layer `abaco` in the annotated dataset. 

# %%
# Reconstruct the dataset using the trained ABaCo model
corrected_dataset = model.correct(seed=42)
# save back to adata
adata.layers['abaco'] = corrected_dataset.set_index(id_col).drop(columns=[batch_col, bio_col]).values
adata.write_h5ad("data/mgnify_tomato.h5ad")

# %% [markdown]
# ## Viewing the ABaCo reconstructed dataset
#
# Here we again take a look at the PCoA but first lets check out the mean differences between the original and batch corrected counts.

# %%
# flatten 
nona = adata.layers['nona'].flatten()
abaco = adata.layers['abaco'].flatten()
# calculate means and diffs
means = (nona + abaco) / 2
diffs = nona - abaco
print(means.shape, diffs.shape)
# mean and limits of agreement
mean_diff = np.mean(diffs)
std_diff = np.std(diffs)
loa_upper = mean_diff + 1.96 * std_diff
loa_lower = mean_diff - 1.96 * std_diff

# creating plot
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=means, y=diffs, 
        mode='markers', 
        marker=dict(color='blue', opacity=0.1)
    )
)
fig.add_hline(y=mean_diff,line_color="black", annotation_text="Mean Diff")
fig.add_hline(y=loa_upper, line_dash="dot", line_color="black")
fig.add_hline(y=loa_lower, line_dash="dot", line_color="black")
# titles
fig.update_layout(
    template="plotly_white",
    title="Bland-Altman Plot: Original vs. ABaCo Corrected",
    xaxis_title="Mean of Original and ABaCo Corrected",
    yaxis_title="Difference (original - abaco)",
)
fig.show()

# %% [markdown]
# No clear tendency for abaco correction to be lower or higher than original counts.

# %%
# Plot the PCoA of the reconstructed dataset
plotPCoA(
    data = corrected_dataset, 
    sample_label=id_col, 
    batch_label=batch_col, 
    experiment_label=bio_col
)

# %% [markdown]
# In the PCoA we see that there is more overlap of the batches based on the colours mixing. 

# %% [markdown]
# ## Conclusion
#
# The goal was to: 
#
# &#x2705; correct the batch effect (reduce clustering of points by colour in the PCoA and pairplot coloured by batches)
#
# A brief visual inspection of the PCoA suggests that ABaCo reduced the batch effect associated with analyzing the samples from different instruments, while still maintaining the original biological variance.
#
#
# ---

# %% [markdown]
#
