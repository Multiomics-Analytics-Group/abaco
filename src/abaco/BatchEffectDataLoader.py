import pandas as pd
import numpy as np
from scipy.stats import gmean
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


def DataPreprocess(path, factors=["sample", "batch", "tissue"]):
    df = pd.read_csv(path)
    df[factors] = df[factors].astype("category")

    return df


def DataTransform(
    data, factors=["sample", "batch", "tissue"], transformation="CLR", count=False
):

    if transformation == "CLR":
        if count == False:
            # Select only OTUs columns and adding a small offset
            df_otu = data.select_dtypes(include="number") + 1e-9

        else:
            df_otu = data.select_dtypes(include="number") + 1

        # Apply CLR transformation to numeric columns
        df_clr = np.log(df_otu.div(gmean(df_otu, axis=1), axis=0))

        # Combine CLR-transformed data with non-numeric columns
        df = pd.concat([data[factors], df_clr], axis=1)

    elif transformation == "Sqrt":
        # Select only OTUs columns
        df_otu = data.select_dtypes(include="number")

        # Apply Square-root transformation to numeric columns
        df_sqrt = np.sqrt(df_otu)

        # Standardize the squared-rooted data
        scaler = StandardScaler()
        df_stsqrt = scaler.fit_transform(df_sqrt)

        # Convert back to DataFrame
        df_stsqrt = pd.DataFrame(df_stsqrt, columns=df_sqrt.columns)

        # Combine data with non-numeric columns
        df = pd.concat([data[factors], df_stsqrt], axis=1)

    elif transformation == "ILR":
        print("Not yet developed")

    elif transformation == "ALR":
        print("To be developed")

    else:
        raise (ValueError(f"Not a valid transformation: {transformation}"))

    return df


def DataReverseTransform(
    data,
    original_data,
    factors=["sample", "batch", "tissue"],
    transformation="CLR",
    count=False,
):

    if transformation == "CLR":
        df_otu = data.select_dtypes(include="number")
        if count == False:
            df_otu_original = original_data.select_dtypes(include="number") + 1e-9

        else:
            df_otu_original = original_data.select_dtypes(include="number") + 1

        df_inv = round(np.exp(df_otu) * gmean(df_otu_original, axis=1)[:, None], 3)

        df = pd.concat([data[factors], df_inv], axis=1)

    return df


def one_hot_encoding(labels):
    # Dictionary of batch labels
    alphabet = labels.unique()
    label_to_int = {label: i for i, label in enumerate(alphabet)}

    # Initialize the one-hot encoded matrix
    one_hot = np.zeros((len(labels), len(alphabet)), dtype=int)

    # Fill the matrix
    for i, label in enumerate(labels):
        if label in label_to_int:
            one_hot[i, label_to_int[label]] = 1

    return torch.tensor(one_hot), alphabet.tolist()


def class_to_int(labels):
    # Dictionary of batch labels
    alphabet = labels.unique()
    label_to_int = {label: i for i, label in enumerate(alphabet)}

    # Initialize the empty array
    classes = np.zeros(len(labels), dtype=int)

    # Fill the matrix
    for i, label in enumerate(labels):
        classes[i] = label_to_int[label]

    return torch.tensor(classes)


def ABaCoDataLoader(
    data,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_label="batch",
    exp_label="tissue",
    batch_size=32,
    total_size=1024,
    total_batch=10,
):

    # Convert data to tensor (structure: tensor([otus], [batch]))
    otu_data = data.select_dtypes(include="number")
    otu_tensor = torch.tensor(otu_data.values, dtype=torch.float32)

    # Add zero padding for input
    n, m = otu_tensor.shape
    zero_padding = torch.zeros((n, total_size - m))

    # Extract labels and convert to one hot encoding matrix
    data_batch = data[batch_label]
    data_tissue = data[exp_label]
    ohe_batch, _ = one_hot_encoding(data_batch)
    ohe_tissue, _ = one_hot_encoding(data_tissue)

    # Add zero padding for batch input
    k, j = ohe_batch.shape
    batch_padding = torch.zeros((k, total_batch - j))

    # Send to device
    otu_tensor = otu_tensor.to(device)
    ohe_batch = ohe_batch.to(device)
    ohe_tissue = ohe_tissue.to(device)
    zero_padding = zero_padding.to(device)
    batch_padding = batch_padding.to(device)

    # otu_dataloader = DataLoader(otu_tensor, batch_size = batch_size)
    # batch_dataloader = DataLoader(ohe_batch, batch_size = batch_size)
    # tissue_dataloader = DataLoader(ohe_tissue, batch_size = batch_size)

    # Defining DataLoader for otus + batch information
    otu_tensor_padded = torch.concat((otu_tensor, zero_padding), 1)
    ohe_batch_padded = torch.concat((ohe_batch, batch_padding), 1)

    otu_batch_tensor = torch.concat((otu_tensor_padded, ohe_batch_padded), 1)
    # otu_batch_dataloader = DataLoader(otu_batch_tensor, batch_size = batch_size)

    # Defining DataLoader for otus + tissue information, also including batch as label for discriminator training
    # otu_tissue_tensor = torch.concat((otu_tensor, ohe_tissue), 1)
    # otu_tissue_dataloader = DataLoader(TensorDataset(otu_tissue_tensor, class_to_int(data_batch)), batch_size = batch_size)

    # Defining DataLoader for otus including tissue as label for classifier training
    # otu_tissue_class_dataloader = DataLoader(TensorDataset(otu_tensor, class_to_int(data_tissue)), batch_size = batch_size)

    # Defining DataLoader for otus including + batch information, also including tissue as label for classificator training
    k_features = torch.full((n,), m, device=device)
    abaco_dataloader = DataLoader(
        TensorDataset(
            otu_batch_tensor, class_to_int(data_tissue).to(device), k_features
        ),
        batch_size=batch_size,
    )
    ohe_dataloader = DataLoader(ohe_tissue, batch_size=batch_size)

    return (
        abaco_dataloader,
        ohe_batch,
        ohe_dataloader,
        otu_data,
        data_batch,
        data_tissue,
    )
