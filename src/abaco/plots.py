import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skbio.stats.ordination import pcoa
from scipy.spatial.distance import pdist, squareform
import plotly.graph_objects as go
from plotly.validator_cache import ValidatorCache
from plotly.subplots import make_subplots
from clustergrammer2 import Network, CGM2
from abaco.dataloader import DataTransform


def plotPCoA(
    data,
    method="aitchison",
    sample_label="sample",
    batch_label="batch",
    experiment_label="tissue",
    mode="base",
):
    """
    Plot Principal Coordinates Analysis (PCoA) for batch effect visualization.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing OTU counts and metadata.
    method : str, optional
        Distance metric to use ('aitchison' or 'bray-curtis'), by default 'aitchison'.
    sample_label : str, optional
        Column name for sample identifiers, by default 'sample'.
    batch_label : str, optional
        Column name for batch identifiers, by default 'batch'.
    experiment_label : str, optional
        Column name for experiment/tissue identifiers, by default 'tissue'.
    mode : str, optional
        Plotting mode ('base' for batch+experiment, 'single' for batch only), by default 'base'.

    Returns
    -------
    None
        Displays a Plotly figure.
    """
    if method == "aitchison":
        # CLR transform
        df = DataTransform(
            data,
            factors=[sample_label, batch_label, experiment_label],
            transformation="CLR",
            count=True,
        )

        # log_transformed = np.log(data.select_dtypes(include = "number") + 1e-9)
        # clr_data = log_transformed - log_transformed.mean(axis=1)[:, None]
        # print(clr_data)

        # Extracting numerical data
        df_otu = df.select_dtypes(include="number")
        # Compute Aitchison distances
        distances = pdist(df_otu, "euclidean")
        distances = squareform(distances)

    elif method == "bray-curtis":
        # Extract numeric data (e.g., OTU count data).
        df_otu = data.select_dtypes(include="number")
        # Convert each sample's counts to relative abundances (row sums are normalized to 1)
        # (Handling potential division by zero)
        row_sums = df_otu.sum(axis=1)
        df_rel = df_otu.div(row_sums.replace(0, np.nan), axis=0).fillna(0)
        # Compute Bray-Curtis distances
        distances = pdist(df_rel, metric="braycurtis")
        distances = squareform(distances)

    else:
        raise (ValueError(f"Method provided not valid: {method}"))

    # PCoA
    pcoa_res = pcoa(distances)
    # Construct DataFrame with principal components and metadata
    df_pcoa = pd.DataFrame(pcoa_res.samples[["PC1", "PC2"]], columns=["PC1", "PC2"])
    df_pcoa.index = (
        data.index
    )  # This assures index are the same and both DataFrames are perfectly aligned
    df_pcoa[[sample_label, batch_label, experiment_label]] = data[
        [sample_label, batch_label, experiment_label]
    ]
    # df_pcoa = pd.concat([data[[sample_label, batch_label, experiment_label]], df_pcoa], axis=1)
    # Extracting available symbols to be used per experiment
    SymbolValidator = ValidatorCache.get_validator("scatter.marker", "symbol")
    raw_symbols = SymbolValidator.values[2::12]

    # Defining a set of colors to be used for batches
    raw_colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "cyan",
        "magenta",
        "yellow",
        "black",
        "brown",
        "pink",
        "gray",
        "olive",
        "teal",
        "navy",
        "maroon",
        "gold",
        "lime",
        "indigo",
        "violet",
        "coral",
        "slateblue",
        "aquamarine",
        "crimson",
        "sienna",
        "salmon",
        "turquoise",
        "lavender",
        "chocolate",
        "tomato",
        "plum",
        "peru",
        "khaki",
        "orchid",
        "springgreen",
        "steelblue",
        "seagreen",
        "darkblue",
        "darkred",
        "darkgreen",
        "darkorange",
        "darkviolet",
        "mediumblue",
        "mediumvioletred",
        "mediumseagreen",
        "midnightblue",
        "lightblue",
        "lightgreen",
        "lightcoral",
        "peachpuff",
    ]

    # Adding symbol to corresponding experiment
    df_pcoa["marker"] = None
    for n, exp in enumerate(df_pcoa[experiment_label].unique()):
        df_pcoa.loc[df_pcoa[experiment_label] == exp, "marker"] = raw_symbols[n]

    # Adding color to corresponding batch
    df_pcoa["color"] = None
    for n, batch in enumerate(df_pcoa[batch_label].unique()):
        df_pcoa.loc[df_pcoa[batch_label] == batch, "color"] = raw_colors[n]

    # Creating the plotly figure
    fig = go.Figure()

    if mode == "base":
        # Creating a for loop to alocate PCA data points per batch
        for batch in df_pcoa[batch_label].unique():
            # Creating a for loop to alocate data points per experiment in the current batch
            for exp in df_pcoa[experiment_label].unique():
                # Ploting the points corresponding to the current batch and tissue
                fig.add_trace(
                    go.Scatter(
                        x=df_pcoa[
                            (df_pcoa[batch_label] == batch)
                            & (df_pcoa[experiment_label] == exp)
                        ]["PC1"],
                        y=df_pcoa[
                            (df_pcoa[batch_label] == batch)
                            & (df_pcoa[experiment_label] == exp)
                        ]["PC2"],
                        marker=dict(
                            color=df_pcoa[
                                (df_pcoa[batch_label] == batch)
                                & (df_pcoa[experiment_label] == exp)
                            ]["color"],
                            size=8,
                        ),
                        marker_symbol=df_pcoa[
                            (df_pcoa[batch_label] == batch)
                            & (df_pcoa[experiment_label] == exp)
                        ]["marker"],
                        legendgroup=batch,
                        legendgrouptitle_text="Batch {}".format(batch),
                        name=exp,
                        mode="markers",
                    )
                )

        # fig.update_layout(xaxis_range = [-5, 5],
        #                  yaxis_range = [-5, 5])

        return fig.show()

    elif mode == "single":
        # Creating a for loop to alocate PCA data points per batch
        for batch in df_pcoa[batch_label].unique():
            # Ploting the points corresponding to the current batch and tissue
            fig.add_trace(
                go.Scatter(
                    x=df_pcoa[(df_pcoa[batch_label] == batch)]["PC1"],
                    y=df_pcoa[(df_pcoa[batch_label] == batch)]["PC2"],
                    marker=dict(
                        color=df_pcoa[(df_pcoa[batch_label] == batch)]["color"],
                        size=8,
                    ),
                    legendgroup=batch,
                    legendgrouptitle_text="Batch {}".format(batch),
                    name=batch,
                    mode="markers",
                )
            )

        # fig.update_layout(xaxis_range = [-5, 5],
        #                  yaxis_range = [-5, 5])

        return fig.show()


def plotPCA(
    data, sample_label="sample", batch_label="batch", experiment_label="tissue"
):
    """
    Plot Principal Component Analysis (PCA) for batch effect visualization.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing OTU counts and metadata.
    sample_label : str, optional
        Column name for sample identifiers, by default 'sample'.
    batch_label : str, optional
        Column name for batch identifiers, by default 'batch'.
    experiment_label : str, optional
        Column name for experiment/tissue identifiers, by default 'tissue'.

    Returns
    -------
    None
        Displays a Plotly figure.
    """
    # Realize the PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data.select_dtypes(include="number"))
    df_pca = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])

    df_pca.index = (
        data.index
    )  # This assures index are the same and both DataFrames are perfectly aligned

    # Add the labels from batch and other important information
    df_pca[[sample_label, batch_label, experiment_label]] = data[
        [sample_label, batch_label, experiment_label]
    ]

    # Extracting available symbols to be used per experiment
    SymbolValidator = ValidatorCache.get_validator("scatter.marker", "symbol")
    raw_symbols = SymbolValidator.values[2::12]

    # Defining a set of colors to be used for batches
    raw_colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "cyan",
        "magenta",
        "yellow",
        "black",
        "brown",
        "pink",
        "gray",
        "olive",
        "teal",
        "navy",
        "maroon",
        "gold",
        "lime",
        "indigo",
        "violet",
        "coral",
        "slateblue",
        "aquamarine",
        "crimson",
        "sienna",
        "salmon",
        "turquoise",
        "lavender",
        "chocolate",
        "tomato",
        "plum",
        "peru",
        "khaki",
        "orchid",
        "springgreen",
        "steelblue",
        "seagreen",
        "darkblue",
        "darkred",
        "darkgreen",
        "darkorange",
        "darkviolet",
        "mediumblue",
        "mediumvioletred",
        "mediumseagreen",
        "midnightblue",
        "lightblue",
        "lightgreen",
        "lightcoral",
        "peachpuff",
    ]

    # Adding symbol to corresponding experiment
    df_pca["marker"] = None
    for n, exp in enumerate(df_pca[experiment_label].unique()):
        df_pca.loc[df_pca[experiment_label] == exp, "marker"] = raw_symbols[n]

    # Adding color to corresponding batch
    df_pca["color"] = None
    for n, batch in enumerate(df_pca[batch_label].unique()):
        df_pca.loc[df_pca[batch_label] == batch, "color"] = raw_colors[n]

    # Creating the plotly figure
    fig = go.Figure()

    # Creating a for loop to alocate PCA data points per batch
    for batch in df_pca[batch_label].unique():
        # Creating a for loop to alocate data points per experiment in the current batch
        for exp in df_pca[experiment_label].unique():
            # Ploting the points corresponding to the current batch and tissue
            fig.add_trace(
                go.Scatter(
                    x=df_pca[
                        (df_pca[batch_label] == batch)
                        & (df_pca[experiment_label] == exp)
                    ]["PC1"],
                    y=df_pca[
                        (df_pca[batch_label] == batch)
                        & (df_pca[experiment_label] == exp)
                    ]["PC2"],
                    marker=dict(
                        color=df_pca[
                            (df_pca[batch_label] == batch)
                            & (df_pca[experiment_label] == exp)
                        ]["color"],
                        size=8,
                    ),
                    marker_symbol=df_pca[
                        (df_pca[batch_label] == batch)
                        & (df_pca[experiment_label] == exp)
                    ]["marker"],
                    legendgroup=batch,
                    legendgrouptitle_text="Batch {}".format(batch),
                    name=exp,
                    mode="markers",
                )
            )

    # fig.update_layout(xaxis_range = [-5, 5],
    #                  yaxis_range = [-5, 5])

    fig.update_layout(legend=dict(font=dict(size=8), itemwidth=30))

    return fig.show()


def plotOTUBox(data, batch_label="batch"):
    """
    Plot boxplots of OTU abundances grouped by batch.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing OTU counts and metadata.
    batch_label : str, optional
        Column name for batch identifiers, by default 'batch'.

    Returns
    -------
    None
        Displays a Plotly figure with dropdown to select OTUs.
    """
    # Extract OTUs columns names
    otu_cols = [col for col in data.columns if col.startswith("OTU")]
    batch_labels = data[batch_label].unique()
    batch_len = len(batch_labels)

    # Converting DataFrame from wide to long
    df_long = pd.melt(
        data,
        id_vars=[batch_label],
        value_vars=otu_cols,
        var_name="OTU",
        value_name="value",
    )

    # Defining a set of colors to be used for batches
    raw_colors = ["blue", "red", "green", "orange", "purple"]

    # Adding color to corresponding batch
    batch_colors = []
    for i in range(batch_len):
        batch_colors.append(raw_colors[i])

    fig = go.Figure()

    # Add traces for each OTU
    for otu in otu_cols:
        for i, batch in enumerate(batch_labels):
            fig.add_trace(
                go.Box(
                    x=df_long[
                        (df_long["OTU"] == otu) & (df_long[batch_label] == batch)
                    ][batch_label],
                    y=df_long[
                        (df_long["OTU"] == otu) & (df_long[batch_label] == batch)
                    ]["value"],
                    marker=dict(
                        color=batch_colors[i]
                    ),  # Apply color to the batch boxplot
                    name=f"Batch {batch}, {otu}",  # Label each trace by the OTU
                    visible=False,  # Set initially to invisible
                )
            )

    # First OTU visible by default
    for i in range(batch_len):
        fig.data[i].visible = True

    # Add dropdown to select which OTU to display
    fig.update_layout(
        xaxis_title="Batch",
        updatemenus=[
            dict(
                buttons=[
                    *[
                        dict(
                            args=[
                                {
                                    "visible": [
                                        (i >= batch_len * idx)
                                        & (i <= batch_len * idx + (batch_len - 1))
                                        for i in range(len(fig.data))
                                    ]
                                }
                            ],  # Toggle visibility
                            label=otu,
                            method="update",
                        )
                        for idx, otu in enumerate(otu_cols)
                    ]
                ],
                direction="down",
                showactive=True,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ],
    )

    return fig.show()


def plotRLE(
    data, sample_label="sample", batch_label="batch", experiment_label="tissue"
):
    """
    Plot Relative Log Expression (RLE) boxplots for each experiment and batch.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing OTU counts and metadata.
    sample_label : str, optional
        Column name for sample identifiers, by default 'sample'.
    batch_label : str, optional
        Column name for batch identifiers, by default 'batch'.
    experiment_label : str, optional
        Column name for experiment/tissue identifiers, by default 'tissue'.

    Returns
    -------
    None
        Displays a Plotly figure with dropdown to select experiments.
    """
    # Extract OTUs column names
    otu_cols = [col for col in data.columns if col.startswith("OTU")]

    # Converting DataFrame from wide to long
    df_long = pd.melt(
        data,
        id_vars=[sample_label, batch_label, experiment_label],
        value_vars=otu_cols,
        var_name="OTU",
        value_name="value",
    )

    # Calculating the medians of each OTU within each experiment
    df_long["medians"] = None
    for OTU in df_long["OTU"].unique():
        for exp in df_long[experiment_label].unique():
            med = np.median(
                df_long[(df_long["OTU"] == OTU) & (df_long[experiment_label] == exp)][
                    "value"
                ]
            )
            df_long.loc[
                (df_long["OTU"] == OTU) & (df_long[experiment_label] == exp), "medians"
            ] = med

    # Incorporating the difference between OTU value in each sample and the median across all samples from the same tissue
    df_long["RLE"] = df_long["value"] - df_long["medians"]

    # Defining a set of colors to be used for batches
    raw_colors = ["blue", "red", "green", "orange", "purple"]

    # Adding color to corresponding batch
    df_long["color"] = None
    for n, batch in enumerate(df_long[batch_label].unique()):
        df_long.loc[df_long[batch_label] == batch, "color"] = raw_colors[n]

    # Generate RLE plots for each experiment
    fig = go.Figure()

    # Add traces for each experiment
    for exp in df_long[experiment_label].unique():
        # Add traces for each batch
        for batch in df_long[batch_label].unique():
            fig.add_trace(
                go.Box(
                    x=df_long[
                        (df_long[experiment_label] == exp)
                        & (df_long[batch_label] == batch)
                    ][sample_label],
                    y=df_long[
                        (df_long[experiment_label] == exp)
                        & (df_long[batch_label] == batch)
                    ]["RLE"],
                    marker_color=df_long[
                        (df_long[experiment_label] == exp)
                        & (df_long[batch_label] == batch)
                    ]["color"].iloc[0],
                    name="Batch {}".format(batch),  # Label each trace by the batch
                    visible=False,  # Set initially to invisible
                )
            )

    # First experiment's traces visible by default
    for i in range(len(df_long[batch_label].unique())):
        fig.data[i].visible = True

    # Add dropdown to select which experiment to display
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    *[
                        dict(
                            args=[
                                {
                                    "visible": [
                                        i // len(df_long[batch_label].unique()) == idx
                                        for i in range(len(fig.data))
                                    ]
                                }
                            ],  # Toggle visibility
                            label=exp,
                            method="update",
                        )
                        for idx, exp in enumerate(df_long[experiment_label].unique())
                    ]
                ],
                direction="down",
                showactive=True,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ]
    )

    # Add horizontal dashed red line at y = 0 as a reference point
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,  # Extend the line across the x-axis
        y0=0,
        y1=0,  # Line positioned at y = 0
        xref="paper",
        yref="y",  # "paper" allows the line to span the entire plot width
        line=dict(color="red", width=2, dash="dash"),  # Dashed red line
    )

    return fig.show()


def plotClusterHeatMap(
    data, batch_label="batch", experiment_label="tissue", sample_label="sample"
):
    """
    Plot a clustered heatmap of scaled OTU data with batch and experiment metadata.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing OTU counts and metadata.
    batch_label : str, optional
        Column name for batch identifiers, by default 'batch'.
    experiment_label : str, optional
        Column name for experiment/tissue identifiers, by default 'tissue'.
    sample_label : str, optional
        Column name for sample identifiers, by default 'sample'.

    Returns
    -------
    clustergrammer2.CGM2Widget
        Clustergrammer2 widget displaying the clustered heatmap.
    """
    # Extracts numerical and categorical data of interest
    data_num = data.select_dtypes(include="number")
    data_num.index = [str(i) for i in data[sample_label]]
    data_cat = data[[batch_label, experiment_label]]
    data_cat.index = [str(i) for i in data[sample_label]]

    # First scaling process - Ensures every observation is scaled according to OTUs
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_num)
    scaled_data = pd.DataFrame(
        scaled_data, columns=data_num.columns, index=data_num.index
    )

    # Second scaling process - Ensures every observation is scaled according to sample
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(scaled_data.T)
    scaled_data = pd.DataFrame(
        scaled_data,
        index=[str(i) for i in data_num.columns],
        columns=[str(i) for i in data_num.index],
    )

    # Create Clustergrammer2 plot
    n2 = Network(CGM2)
    n2.load_df(scaled_data, meta_col=data_cat)
    n2.cluster()
    return n2.widget()


def plot_LISI_perplexity(
    df_c,
    df_i,
    n_samples: int,
    x_col: str = "perplexity",
    y_col_c: str = "cLISI",
    y_col_i: str = "iLISI",
    title_c: str = "Biological conservation (cLISI)",
    title_i: str = "Batch mixing (iLISI)",
):
    """
    Plot cLISI and iLISI scores as a function of perplexity.

    Parameters
    ----------
    df_c : pandas.DataFrame
        DataFrame containing cLISI scores and perplexity values.
    df_i : pandas.DataFrame
        DataFrame containing iLISI scores and perplexity values.
    n_samples : int
        Number of samples in the dataset.
    x_col : str, optional
        Column name for perplexity values, by default 'perplexity'.
    y_col_c : str, optional
        Column name for cLISI scores, by default 'cLISI'.
    y_col_i : str, optional
        Column name for iLISI scores, by default 'iLISI'.
    title_c : str, optional
        Title for the cLISI subplot, by default 'Biological conservation (cLISI)'.
    title_i : str, optional
        Title for the iLISI subplot, by default 'Batch mixing (iLISI)'.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        Plotly figure with cLISI and iLISI subplots.
    """
    ideal_k = max(int(np.sqrt(n_samples)), 1)

    # x-axis limits
    xmin = min(df_c[x_col].min(), df_i[x_col].min()) - 0.5
    xmax = max(df_c[x_col].max(), df_i[x_col].max()) + 0.5

    # y-intersection
    y_c = np.interp(ideal_k, df_c[x_col].values, df_c[y_col_c].values)
    y_i = np.interp(ideal_k, df_i[x_col].values, df_i[y_col_i].values)

    # Create 1×2 subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=False,
        shared_xaxes=True,
        subplot_titles=(title_c, title_i),
    )

    # cLISI trace
    fig.add_trace(
        go.Scatter(
            x=df_c[x_col],
            y=df_c[y_col_c],
            mode="lines+markers",
            name="cLISI",
            line=dict(color="green"),
            marker=dict(color="green"),
        ),
        row=1,
        col=1,
    )
    # ideal point in cLISI
    fig.add_shape(
        dict(
            type="line",
            x0=ideal_k,
            y0=0,
            x1=ideal_k,
            y1=y_c,
            line=dict(color="red", dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_shape(
        dict(
            type="line",
            x0=xmin,
            y0=y_c,
            x1=ideal_k,
            y1=y_c,
            line=dict(color="red", dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_annotation(
        dict(
            x=ideal_k,
            y=y_c,
            xref="x1",
            yref="y1",
            text=f"{y_c:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=30,
            ay=-30,
            font=dict(color="red"),
        )
    )

    # iLISI trace
    fig.add_trace(
        go.Scatter(
            x=df_i[x_col],
            y=df_i[y_col_i],
            mode="lines+markers",
            name="iLISI",
            line=dict(color="blue"),
            marker=dict(color="blue"),
        ),
        row=1,
        col=2,
    )
    # Ideal line in iLISI
    fig.add_shape(
        dict(
            type="line",
            x0=ideal_k,
            y0=0,
            x1=ideal_k,
            y1=y_i,
            line=dict(color="red", dash="dash"),
        ),
        row=1,
        col=2,
    )
    fig.add_shape(
        dict(
            type="line",
            x0=xmin,
            y0=y_i,
            x1=ideal_k,
            y1=y_i,
            line=dict(color="red", dash="dash"),
        ),
        row=1,
        col=2,
    )
    fig.add_annotation(
        dict(
            x=ideal_k,
            y=y_i,
            xref="x2",
            yref="y2",
            text=f"{y_i:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=30,
            ay=-30,
            font=dict(color="red"),
        )
    )

    # Update axes
    fig.update_xaxes(title_text="Perplexity (k)", range=[xmin, xmax], row=1, col=1)
    fig.update_xaxes(title_text="Perplexity (k)", range=[xmin, xmax], row=1, col=2)
    fig.update_yaxes(title_text="Normalized cLISI", row=1, col=1)
    fig.update_yaxes(title_text="Normalized iLISI", row=1, col=2)

    fig.update_layout(template="plotly_white", showlegend=False)
    return fig
