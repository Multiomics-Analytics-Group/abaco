import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.cluster import KMeans

from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix, permanova

from abaco.BatchEffectDataLoader import DataTransform


def kBET(data, batch_label="batch"):

    data_otus = data.select_dtypes(include="number")
    data_batch = data[batch_label]

    n = len(data_batch)
    k = int(np.sqrt(data_otus.shape[0]))  # k-nearest neighbors
    dof = len(data_batch.unique()) - 1  # Degrees of freedom

    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(data_otus)

    _, indx = knn.kneighbors(data_otus)

    p_values = []
    for j, neighbor in enumerate(indx):
        gamma_j = []
        for i in data_batch.unique():
            mu_ij = (
                round(sum(data_batch == i) / n * k + 1e-6) + 1e-6
            )  # Expected number of samples from batch i in neighbor j
            n_ij = 0  # Actual number of samples from batch i in neighbor j

            for s in neighbor:
                if data_batch.iloc[s] == i:
                    n_ij += 1  # Identifies batch label on neighborhood

            gamma_ij = (n_ij - mu_ij) ** 2 / mu_ij
            gamma_j.append(gamma_ij)

        sum_gamma_j = sum(gamma_j)  # Follows Chi-squared distribution

        p_j = 1 - chi2.cdf(sum_gamma_j, dof)  # Pearson Chi-squared test
        p_values.append(p_j)

    return sum(1 for p in p_values if p > 0.05) / n


def ASW(data, interest_label="tissue"):

    average_silhouette = silhouette_score(
        data.select_dtypes(include="number"), data[interest_label]
    )

    return average_silhouette


def ARI(data, interest_label="tissue", n_clusters=None):

    data_otus = data.select_dtypes(include="number")  # OTUs
    data_bio = data[interest_label]  # Labels

    if n_clusters == None:
        kmeans = KMeans(
            n_clusters=len(set(data_bio)), random_state=42
        )  # KMeans clustering
    else:
        kmeans = KMeans(
            n_clusters=n_clusters, random_state=42
        )  # KMeans clustering w/ n clusters

    predicted_clusters = kmeans.fit_predict(data_otus)  # Predicting label of cluster

    ari = adjusted_rand_score(data_bio, predicted_clusters)  # ARI
    return ari


# NMI: Normalized Mutual Information
def NMI(data, bio_label, n_cluster=None, average_method="arithmetic"):
    taxa_matrix = data.select_dtypes(include="number")
    true_labels = data[bio_label]

    # Decide number of clusters
    k = n_cluster if n_cluster is not None else len(true_labels.unique())
    kmeans = KMeans(n_clusters=k, random_state=42)
    pred_labels = kmeans.fit_predict(taxa_matrix)

    # Compute NMI
    nmi_score = normalized_mutual_info_score(
        true_labels, pred_labels, average_method=average_method
    )
    return nmi_score


def all_metrics(data, bio_label, batch_label, n_cluster=None):
    """
    Perform all metrics to address batch correction and biological conservation.

    Batch correction:
        kBET
        ARI (batch)
        ASW (batch)
    Biological conservation:
        NMI
        ARI (bio)
        ASW (bio)
    """
    # Batch correction metrics
    kbet = kBET(data, batch_label=batch_label)
    ari_batch = 1 - ARI(data, interest_label=batch_label, n_clusters=n_cluster)
    asw_batch = 1 - ASW(data, interest_label=batch_label)

    # Biological conservation metrics
    nmi = NMI(data, bio_label=bio_label)
    ari_bio = ARI(data, interest_label=bio_label, n_clusters=n_cluster)
    asw_bio = ASW(data, interest_label=bio_label)

    # Return dictionary with all values
    batch_metrics = {"kBET": kbet, "ARI": ari_batch, "ASW": asw_batch}
    bio_metrics = {"NMI": nmi, "ARI": ari_bio, "ASW": asw_bio}

    return batch_metrics, bio_metrics


# New reviewd functions
# cLISI = iLISI but the labels are just interchanged


def cLISI_raw(
    data: pd.DataFrame,
    bio_label: str,
    k: int = None,
):
    """
    Computes non-normalized cell-type LISI.

    Parameters:
        data: [pd.DataFrame]
            DataFrame containing normalized numeric type taxonomic groups, and any other column category as factor-type values.
        bio_label: [str]
            Name of the column in data with the categorical bio-type labels.
        k: [int]
            Optional, neighborhood size. By default it is the square root of the total number of samples.

    Returns:
        cLISI: [float]
            Mean raw cLISI across all samples. Goes from 1 to the number of unique bio-type labels.
    """

    # Extract numerico matrix and labels
    data_otus = data.select_dtypes(include="number").values
    data_bio = data[bio_label].values
    n_samples = data_otus.shape[0]

    # Determine k
    if k is None:
        k = max(1, int(np.sqrt(n_samples)))
    # Build kNN graph in Euclidean space
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(data_otus)
    _, neighbors = knn.kneighbors(data_otus, return_distance=True)

    # Compute per-biological type inverse Simpson's index
    isi_list = []
    for idx in neighbors:
        # Count frequency of each label in the neighborhood
        counts = pd.Series(data_bio[idx]).value_counts().values
        # Get probability of label in the neighborhood
        probs = counts / counts.sum()
        # Simpson's diversity: sum of squared probabilities
        si = np.sum(probs**2)
        # Inverse simpson's index
        isi = 1.0 / si
        isi_list.append(isi)

    # Return the average raw cLISI
    cLISI = np.mean(isi_list)
    return cLISI


def cLISI_norm(
    data: pd.DataFrame,
    bio_label: str,
    k: int = None,
    n_bio=None,
):
    """
    Computes ranked normalized cell-type LISI in [0, 1]

    Parameters:
        data: [pd.DataFrame]
            DataFrame containing normalized numeric type taxonomic groups, and any other column category as factor-type values.
        bio_label: [str]
            Name of the column in data with the categorical bio-type labels.
        k: [int]
            Optional, neighborhood size. By default it is the square root of the total number of samples.
        n_bio: [int]
            Optional, number of biological labels. If not provided the function will extract it from unique annotations from the data.

    Returns:
        clisi_norm: [float]
            Mean normalized cLISI across all samples. Goes from 0 to 1.
    """
    # Compute raw cLISI score
    raw = cLISI_raw(data=data, bio_label=bio_label, k=k)
    # Number of distinct bio-type labels
    if n_bio is None:
        n_bio = data[bio_label].nunique()
    # In case there is only 1 biological group
    if n_bio < 2:
        return 1.0
    # Normalized rank from 0 to 1
    clisi_norm = (n_bio - raw) / (n_bio - 1)
    return clisi_norm


def cLISI_full_rank(
    data: pd.DataFrame,
    bio_label: str,
    n_bio=None,
    perplexities: list = None,
):
    """
    Computes ranked normalized cell-type LISI within all possible perplexity range.

    Parameters:
        data: [pd.DataFrame]
            DataFrame containing normalized numeric type taxonomic groups, and any other column category as factor-type values.
        bio_label: [str]
            Name of the column in data with the categorical bio-type labels.
        n_bio: [int]
            Optional, number of biological labels. If not provided the function will extract it from unique annotations from the data.
        perplexities: [list]
            Optional, possible perplexities (k) values. If not provided the funciton will use all possible k values.

    Returns:
        clisi_table: [pd.DataFrame]
            DataFrame with all cLISI values for all perplexities.
    """
    # Define perplexities to be used
    if perplexities is None:
        perplexities = [i + 1 for i in range(data.shape[0])]

    clisis = []
    for k in perplexities:
        clisi = cLISI_norm(data=data, bio_label=bio_label, k=k)
        clisis.append({"perplexity": k, "cLISI": clisi})
    clisi_table = pd.DataFrame(clisis)
    return clisi_table


def iLISI_raw(
    data: pd.DataFrame,
    batch_label: str,
    k: int = None,
):
    """
    Computes non-normalized batch-mixing LISI.

    Parameters:
        data: [pd.DataFrame]
            DataFrame containing normalized numeric type taxonomic groups, and any other column category as factor-type values.
        batch_label: [str]
            Name of the column in data with the categorical batch-type labels.
        k: [int]
            Optional, neighborhood size. By default it is the square root of the total number of samples.

    Returns:
        iLISI: [float]
            Mean raw iLISI across all samples. Goes from 1 to the number of unique batch-type labels.
    """

    # Extract numerico matrix and labels
    data_otus = data.select_dtypes(include="number").values
    data_batch = data[batch_label].values
    n_samples = data_otus.shape[0]

    # Determine k
    if k is None:
        k = max(1, int(np.sqrt(n_samples)))
    # Build kNN graph in Euclidean space
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(data_otus)
    _, neighbors = knn.kneighbors(data_otus, return_distance=True)

    # Compute per-biological type inverse Simpson's index
    isi_list = []
    for idx in neighbors:
        # Count frequency of each label in the neighborhood
        counts = pd.Series(data_batch[idx]).value_counts().values
        # Get probability of label in the neighborhood
        probs = counts / counts.sum()
        # Simpson's diversity: sum of squared probabilities
        si = np.sum(probs**2)
        # Inverse simpson's index
        isi = 1.0 / si
        isi_list.append(isi)

    # Return the average raw cLISI
    iLISI = np.mean(isi_list)
    return iLISI


def iLISI_norm(
    data: pd.DataFrame,
    batch_label: str,
    k: int = None,
    n_batch: int = None,
):
    """
    Computes ranked normalized cell-type LISI in [0, 1]

    Parameters:
        data: [pd.DataFrame]
            DataFrame containing normalized numeric type taxonomic groups, and any other column category as factor-type values.
        batch_label: [str]
            Name of the column in data with the categorical batch-type labels.
        k: [int]
            Optional, neighborhood size. By default it is the square root of the total number of samples.
        n_batch: [int]
            Optional, number of batch labels. If not provided the function will extract it from unique annotations from the data.

    Returns:
        ilisi_norm: [float]
            Mean normalized iLISI across all samples. Goes from 0 to 1.
    """
    # Compute raw iLISI score
    raw = iLISI_raw(data=data, batch_label=batch_label, k=k)
    # Number of distinct batch-type labels
    if n_batch is None:
        n_batch = data[batch_label].nunique()
    # In case there is only 1 batch group
    if n_batch < 2:
        return 1.0
    # Normalized rank from 0 to 1
    ilisi_norm = (raw - 1) / (n_batch - 1)
    return ilisi_norm


def iLISI_full_rank(
    data: pd.DataFrame,
    batch_label: str,
    n_batch=None,
    perplexities: list = None,
):
    """
    Computes ranked normalized batch-mixing LISI within all possible perplexity range.

    Parameters:
        data: [pd.DataFrame]
            DataFrame containing normalized numeric type taxonomic groups, and any other column category as factor-type values.
        batch_label: [str]
            Name of the column in data with the categorical batch-type labels.
        n_batch: [int]
            Optional, number of batch labels. If not provided the function will extract it from unique annotations from the data.
        perplexities: [list]
            Optional, possible perplexities (k) values. If not provided the funciton will use all possible k values.

    Returns:
        ilisi_table: [pd.DataFrame]
            DataFrame with all iLISI values for all perplexities.
    """
    # Define perplexities to be used
    if perplexities is None:
        perplexities = [i + 1 for i in range(data.shape[0])]

    ilisis = []
    for k in perplexities:
        ilisi = iLISI_norm(data=data, batch_label=batch_label, k=k)
        ilisis.append({"perplexity": k, "iLISI": ilisi})
    ilisi_table = pd.DataFrame(ilisis)
    return ilisi_table


# Pairwise distance
def pairwise_distance(data, sample_label, batch_label, bio_label):
    # Step 1: normalized data to compute euclidean-like distance
    norm_data = DataTransform(
        data,
        factors=[sample_label, batch_label, bio_label],
        transformation="CLR",
        count=True,
    )
    norm_counts = norm_data.select_dtypes(include="number").values
    sample_ids = norm_data[sample_label].values
    bios = norm_data[bio_label].values

    # Step 2: compute pairwise distance (euclidean) to every point
    pair_dists = squareform(pdist(norm_counts, metric="euclidean"))
    pair_dists_df = (
        pd.DataFrame(
            pair_dists, index=norm_data[sample_label], columns=norm_data[sample_label]
        )
        .reset_index()
        .rename(columns={sample_label: "pointA"})
    )

    # Get longer dataframe
    long_dists_df = pair_dists_df.melt(
        id_vars="pointA", var_name="pointB", value_name="distance"
    )

    # Remove distances to the same point
    long_dists_df = long_dists_df.query("pointA != pointB")

    # Add biological group information
    bio_map = dict(zip(sample_ids, bios))
    long_dists_df["pointA_bio"] = long_dists_df["pointA"].map(bio_map)
    long_dists_df["pointB_bio"] = long_dists_df["pointB"].map(bio_map)

    # Step 3: compute pairwise distance within and between every group
    within_bios = []
    between_bios = []
    for idx, point in long_dists_df.iterrows():
        if point["pointA_bio"] == point["pointB_bio"]:
            within_bios.append(point["distance"])
        else:
            between_bios.append(point["distance"])

    within_bios = np.array(within_bios)
    between_bios = np.array(between_bios)

    # Step 5: summarize
    mean_all_dists = long_dists_df["distance"].mean()
    mean_within_dists = within_bios.mean()
    mean_between_dists = between_bios.mean()

    return mean_all_dists, mean_within_dists, mean_between_dists


def pairwise_distance_std(data, sample_label, batch_label, bio_label):
    # Step 1: normalized data to compute euclidean-like distance
    norm_data = DataTransform(
        data,
        factors=[sample_label, batch_label, bio_label],
        transformation="CLR",
        count=True,
    )
    norm_counts = norm_data.select_dtypes(include="number").values
    sample_ids = norm_data[sample_label].values
    bios = norm_data[bio_label].values

    # Step 2: compute pairwise distance (euclidean) to every point
    pair_dists = squareform(pdist(norm_counts, metric="euclidean"))
    pair_dists_df = (
        pd.DataFrame(
            pair_dists, index=norm_data[sample_label], columns=norm_data[sample_label]
        )
        .reset_index()
        .rename(columns={sample_label: "pointA"})
    )

    # Get longer dataframe
    long_dists_df = pair_dists_df.melt(
        id_vars="pointA", var_name="pointB", value_name="distance"
    )

    # Remove distances to the same point
    long_dists_df = long_dists_df.query("pointA != pointB")

    # Add biological group information
    bio_map = dict(zip(sample_ids, bios))
    long_dists_df["pointA_bio"] = long_dists_df["pointA"].map(bio_map)
    long_dists_df["pointB_bio"] = long_dists_df["pointB"].map(bio_map)

    # Step 3: compute pairwise distance within and between every group
    within_bios = []
    between_bios = []
    for idx, point in long_dists_df.iterrows():
        if point["pointA_bio"] == point["pointB_bio"]:
            within_bios.append(point["distance"])
        else:
            between_bios.append(point["distance"])

    within_bios = np.array(within_bios)
    between_bios = np.array(between_bios)

    # Step 5: summarize
    mean_all_dists = long_dists_df["distance"].std()
    mean_within_dists = within_bios.std()
    mean_between_dists = between_bios.std()

    return mean_all_dists, mean_within_dists, mean_between_dists


def pairwise_distance_multi_run(data, sample_label, batch_label, bio_label):
    # Step 1: normalized data to compute euclidean-like distance
    norm_data = DataTransform(
        data,
        factors=[sample_label, batch_label, bio_label],
        transformation="CLR",
        count=True,
    )
    norm_counts = norm_data.select_dtypes(include="number").values
    sample_ids = norm_data[sample_label].values
    bios = norm_data[bio_label].values

    # Step 2: compute pairwise distance (euclidean) to every point
    pair_dists = squareform(pdist(norm_counts, metric="euclidean"))
    pair_dists_df = (
        pd.DataFrame(
            pair_dists, index=norm_data[sample_label], columns=norm_data[sample_label]
        )
        .reset_index()
        .rename(columns={sample_label: "pointA"})
    )

    # Get longer dataframe
    long_dists_df = pair_dists_df.melt(
        id_vars="pointA", var_name="pointB", value_name="distance"
    )

    # Remove distances to the same point
    long_dists_df = long_dists_df.query("pointA != pointB")

    # Add biological group information
    bio_map = dict(zip(sample_ids, bios))
    long_dists_df["pointA_bio"] = long_dists_df["pointA"].map(bio_map)
    long_dists_df["pointB_bio"] = long_dists_df["pointB"].map(bio_map)

    return long_dists_df


def PERMANOVA(data, sample_label, batch_label, bio_label):

    samples = data[sample_label].values
    bios = data[bio_label].values
    counts = data.select_dtypes(include="number").values + 1e-6

    # Compute Bray-Curtis distance matrix
    bray = pdist(counts, metric="braycurtis")
    dist_mat = squareform(bray)
    dm = DistanceMatrix(dist_mat, ids=samples)

    # PERMANOVA - Bray-curtis
    res_bc = permanova(distance_matrix=dm, grouping=bios)

    # Compute manually R^2 = F(k-1) / (F(k - 1) + (N-k)) , k = number of groups, N = number of samples, F = F-statistic (test statistic)
    res_bc["R2"] = (
        res_bc["test statistic"]
        * (len(bios.unique()) - 1)
        / (
            res_bc["test statistic"] * (len(bios.unique()) - 1)
            + (len(samples) - len(bios.unique()))
        )
    )

    # Compute Aitchison distance matrix
    norm_data = DataTransform(
        data,
        factors=[sample_label, batch_label, bio_label],
        transformation="CLR",
        count=True,
    )
    norm_counts = norm_data.select_dtypes(include="number").values
    aitch = pdist(norm_counts, metric="euclidean")
    dist_mat = squareform(aitch)
    dm = DistanceMatrix(dist_mat, ids=samples)

    # PERMANOVA - Aitchison
    res_ait = permanova(distance_matrix=dm, grouping=bios)

    # Compute manually R^2 = F(k-1) / (F(k - 1) + (N-k)) , k = number of groups, N = number of samples, F = F-statistic (test statistic)
    res_ait["R2"] = (
        res_ait["test statistic"]
        * (len(bios.unique()) - 1)
        / (
            res_ait["test statistic"] * (len(bios.unique()) - 1)
            + (len(samples) - len(bios.unique()))
        )
    )

    return res_bc, res_ait
