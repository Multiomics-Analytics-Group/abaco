import pandas as pd
import numpy as np
from combat.pycombat import pycombat
from inmoose.pycombat import pycombat_seq
import statsmodels.api as sm
from sklearn.preprocessing import scale
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.base import TransformerMixin, BaseEstimator


# Batch Mean Centering
def correctBMC(data, sample_label, batch_label, exp_label):
    """
    This function, LITERALLY, substracts the mean of each batch (group) from each feature.
    Perform Batch Mean Centering (BMC) correction.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing OTU counts and metadata.
    sample_label : str
        Column name for sample identifiers.
    batch_label : str
        Column name for batch identifiers.
    exp_label : str
        Column name for experiment/tissue identifiers.

    Returns
    -------
    pandas.DataFrame
        DataFrame with sample, experiment, batch, and batch mean centered features.
    """
    features = data.select_dtypes(include="number")
    batch_means = features.groupby(data[batch_label]).transform("mean")
    corrected = features - batch_means
    df_all = pd.concat(
        [data[sample_label], data[exp_label], data[batch_label], corrected], axis=1
    )
    return df_all


# ComBat
def correctCombat(
    data, sample_label="sample", batch_label="batch", experiment_label="tissue"
):
    """
    Perform ComBat batch correction.

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
    pandas.DataFrame
        DataFrame with sample, batch, experiment, and ComBat-corrected features.
    """
    num_data = data.select_dtypes(include="number")
    batch_data = [batch for batch in data[batch_label]]
    cov_data = [exp for exp in data[experiment_label]]

    corrected_data = pycombat(num_data.T, batch_data, cov_data)
    data_combat = pd.concat(
        [data[[sample_label, batch_label, experiment_label]], corrected_data.T], axis=1
    )

    return data_combat


# Limma (removeBatchEffect)
def correctLimma_rBE(
    data, sample_label="sample", batch_label="batch", covariates_labels=None
):
    """
    Perform batch correction using Limma's removeBatchEffect approach.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing OTU counts and metadata.
    sample_label : str, optional
        Column name for sample identifiers, by default 'sample'.
    batch_label : str, optional
        Column name for batch identifiers, by default 'batch'.
    covariates_labels : str or list of str, optional
        Additional covariate column(s) to include in the model.

    Returns
    -------
    pandas.DataFrame
        DataFrame with original labels and batch-corrected numeric data.
    """
    # Extract numeric variables from data
    num_data = data.select_dtypes(include="number")

    # Convert batch labels to one-hot encoded DataFrame for regression;
    # drop_first=True to avoid multicollinearity but keep all resulting column names intact.
    batch = pd.get_dummies(data[batch_label], drop_first=True)

    # Combine batch and covariates if provided
    if covariates_labels is not None:
        # Get dummy variables for covariates (if more than one, this returns multiple columns)
        covariates = pd.get_dummies(data[covariates_labels], drop_first=True)
        # Build the full design matrix (batch effects + covariates)
        design_matrix = pd.concat([batch, covariates], axis=1)

        # For the batch-only design matrix, keep the batch part and fill zeros for covariates
        zeros_cov = pd.DataFrame(
            np.zeros_like(covariates),
            columns=covariates.columns,
            index=covariates.index,
        )
        design_matrix_batch = pd.concat([batch, zeros_cov], axis=1)
        # Add constant to the design matrix
        design_matrix_batch = sm.add_constant(design_matrix_batch, has_constant="add")
        design_matrix_batch = design_matrix_batch.astype(float)
    else:
        design_matrix = batch.copy()
        design_matrix_batch = design_matrix.copy()

    # Ensure an intercept (constant) is added to the full design matrix
    design_matrix = sm.add_constant(design_matrix, has_constant="add")
    design_matrix = design_matrix.astype(float)

    # Initialize a DataFrame to store batch-corrected values
    corrected_data = pd.DataFrame(index=num_data.index, columns=num_data.columns)

    # Regress out batch effect for each feature
    for feature in num_data.columns:
        model = sm.OLS(num_data[feature], design_matrix).fit()

        # Predict the portion attributable to batch effects only using the design_matrix_batch
        batch_effect = model.predict(design_matrix_batch)

        # Subtract the estimated batch effect from the original feature values
        corrected_data[feature] = num_data[feature] - batch_effect

    # Prepare final DataFrame with original labels and corrected numeric data.
    # If there are covariates, include them as well.
    columns_to_keep = [sample_label, batch_label]
    if covariates_labels is not None:
        if isinstance(covariates_labels, list):
            columns_to_keep += covariates_labels
        else:
            columns_to_keep.append(covariates_labels)

    # Combine the metadata with the corrected numeric data.
    data_limma = pd.concat(
        [
            data[columns_to_keep].reset_index(drop=True),
            corrected_data.reset_index(drop=True),
        ],
        axis=1,
    )

    return data_limma


# PLSDA-batch
def correctPLSDAbatch(
    df: pd.DataFrame,
    sample_label: str,
    exp_label: str,
    batch_label: str,
    ncomp_trt: int = 1,
    ncomp_batch: int = 1,
):
    """
    Perform PLSDA-batch correction.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing OTU counts and metadata.
    sample_label : str
        Column name for sample identifiers.
    exp_label : str
        Column name for experiment/tissue identifiers.
    batch_label : str
        Column name for batch identifiers.
    ncomp_trt : int, optional
        Number of treatment components, by default 1.
    ncomp_batch : int, optional
        Number of batch components, by default 1.

    Returns
    -------
    pandas.DataFrame
        DataFrame with sample, experiment, batch, and PLSDA-batch corrected features.
    """
    y_sample = df[sample_label]
    y_trt = df[exp_label]
    y_batch = df[batch_label]
    df = df.select_dtypes(include="number")
    X = df.values  # (n, p)
    # Step 1: Encode outcomes
    Y_trt = pd.get_dummies(y_trt).values  # (n, n_trt)
    Y_batch = pd.get_dummies(y_batch).values  # (n, n_batch)

    # Step 2: Fit Partial Least Squares for treatment
    pls_trt = PLSRegression(n_components=ncomp_trt, scale=True)
    pls_trt.fit(X, Y_trt)
    T_trt = pls_trt.x_scores_  # (n, ncomp_trt)
    P_trt = pls_trt.x_loadings_  # (p, ncomp_trt)

    # Step 3: Deflate X by treatment components
    X_res = X - T_trt @ P_trt.T  # (n, p)

    # Step 4: Fit Partial Least Squares for batch on residuals
    pls_batch = PLSRegression(n_components=ncomp_batch, scale=True)
    pls_batch.fit(X_res, Y_batch)
    T_batch = pls_batch.x_scores_  # (n, ncomp_trt)
    P_batch = pls_batch.x_loadings_  # (p, ncomp_trt)

    # Step 5: Substract batch variation from orinial X
    X_nobatch = X - T_batch @ P_batch.T

    # Return as pd.DataFrame
    df_corrected = pd.DataFrame(
        X_nobatch,
        index=df.index,
        columns=df.columns,
    )
    df_all = pd.concat([y_trt, y_batch, y_sample, df_corrected], axis=1)
    return df_all


# PLSDA-batch analogous to R implementation
def deflate_mtx(X, t):
    """
    Deflate matrix X by component t: X - t (t^T t)^{-1} t^T X

    Parameters
    ----------
    X : numpy.ndarray
        Data matrix to be deflated.
    t : numpy.ndarray
        Component vector.

    Returns
    -------
    numpy.ndarray
        Deflated matrix.
    """
    # t: (n_samples,)
    denom = t.T @ t
    if denom == 0:
        return X.copy()
    proj = np.outer(t, (t.T @ X) / denom)
    return X - proj


class PLSDA:
    """
    Partial Least Squares Discriminant Analysis (PLSDA) implementation.

    Parameters
    ----------
    ncomp : int, optional
        Number of components to extract, by default 1.
    keepX : list of int, optional
        Number of variables to keep for each component (for sparsity).
    tol : float, optional
        Convergence tolerance, by default 1e-6.
    max_iter : int, optional
        Maximum number of iterations, by default 500.

    Attributes
    ----------
    t_ : numpy.ndarray
        X scores.
    u_ : numpy.ndarray
        Y scores.
    a_ : numpy.ndarray
        X loadings.
    b_ : numpy.ndarray
        Y loadings.
    iters_ : list
        Number of iterations per component.
    exp_var_ : list
        Explained variance per component.
    """

    def __init__(self, ncomp=1, keepX=None, tol=1e-6, max_iter=500):
        self.ncomp = ncomp
        self.keepX = keepX or []
        self.t_ = None
        self.u_ = None
        self.a_ = None
        self.b_ = None
        self.iters_ = []
        self.exp_var_ = None
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, Y):
        # X, Y are centered and scaled numpy arrays
        n, p = X.shape
        _, q = Y.shape
        keepX = self.keepX if len(self.keepX) == self.ncomp else [p] * self.ncomp
        T = np.zeros((n, self.ncomp))
        U = np.zeros((n, self.ncomp))
        A = np.zeros((p, self.ncomp))
        B = np.zeros((q, self.ncomp))
        X_temp = X.copy()
        Y_temp = Y.copy()
        for h in range(self.ncomp):
            # initial SVD-based loadings
            M = X_temp.T @ Y_temp
            u, s, vh = np.linalg.svd(M, full_matrices=False)
            a = u[:, 0]
            b = vh.T[:, 0]
            # normalize loadings
            a = a / np.linalg.norm(a)
            b = b / np.linalg.norm(b)
            t = X_temp @ a
            u_comp = Y_temp @ b
            # iterative NIPALS
            it = 0
            while it < self.max_iter:
                it += 1
                # update a
                a_new = X_temp.T @ u_comp
                if keepX[h] < p:
                    # sparsity: zero smallest abs entries
                    abs_a = np.abs(a_new)
                    thresh = np.sort(abs_a)[-keepX[h]] if keepX[h] > 0 else np.inf
                    a_new[abs_a < thresh] = 0
                a_new = a_new / np.linalg.norm(a_new)
                t = X_temp @ a_new
                b_new = Y_temp.T @ t
                b_new = b_new / np.linalg.norm(b_new)
                u_comp = Y_temp @ b_new
                if np.linalg.norm(a_new - a) < self.tol:
                    break
                a, b = a_new, b_new
            # store component
            T[:, h] = t
            U[:, h] = u_comp
            A[:, h] = a_new
            B[:, h] = b_new
            self.iters_.append(it)
            # deflate
            X_temp = deflate_mtx(X_temp, t)
            Y_temp = deflate_mtx(Y_temp, u_comp)
        self.t_ = T
        self.u_ = U
        self.a_ = A
        self.b_ = B
        # explained variance on X
        tot_var = np.sum(X**2)
        var_expl = [
            np.sum((T[:, i : i + 1] @ A[:, i : i + 1].T) ** 2) / tot_var
            for i in range(self.ncomp)
        ]
        self.exp_var_ = var_expl
        return self


def correctPLSDAbatch_R(
    df,
    sample_label,
    exp_label,
    batch_label,
    ncomp_trt=1,
    ncomp_bat=1,
    keepX_trt=None,
    keepX_bat=None,
    tol=1e-6,
    max_iter=500,
    near_zero_var=True,
    balance=True,
):
    """
    Python adaptation of PLSDA_batch from R. Returns corrected DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing OTU counts and metadata.
    sample_label : str
        Column name for sample identifiers.
    exp_label : str
        Column name for experiment/tissue identifiers.
    batch_label : str
        Column name for batch identifiers.
    ncomp_trt : int, optional
        Number of treatment components, by default 1.
    ncomp_bat : int, optional
        Number of batch components, by default 1.
    keepX_trt : list of int, optional
        Number of variables to keep for each treatment component.
    keepX_bat : list of int, optional
        Number of variables to keep for each batch component.
    tol : float, optional
        Convergence tolerance, by default 1e-6.
    max_iter : int, optional
        Maximum number of iterations, by default 500.
    near_zero_var : bool, optional
        Whether to filter near-zero variance features, by default True.
    balance : bool, optional
        Whether to balance design, by default True.

    Returns
    -------
    pandas.DataFrame
        DataFrame with sample, experiment, batch, and corrected features.
    """
    y_sample = df[sample_label]
    y_trt = df[exp_label]
    y_batch = df[batch_label]
    df = df.select_dtypes(include="number")
    X = df.values.copy()
    n, p = X.shape
    # near zero variance filtering
    if near_zero_var:
        var = X.var(axis=0)
        keep = var > 1e-8
        X = X[:, keep]
        cols = df.columns[keep]
    else:
        cols = df.columns
    # encode Y
    Y_trt = pd.get_dummies(y_trt).values
    Y_bat = pd.get_dummies(y_batch).values
    # weighting
    weight = np.ones(n)
    if not balance:
        # implement weighted design (omitted for brevity)
        pass
    # scale
    Xs = scale(X, with_mean=True, with_std=True)
    Ys_trt = scale(weight[:, None] * Y_trt)
    Ys_bat = scale(Y_bat)
    # stage1: treatment
    pls_trt = PLSDA(
        ncomp=ncomp_trt, keepX=keepX_trt or [p] * ncomp_trt, tol=tol, max_iter=max_iter
    )
    pls_trt.fit(Xs, Ys_trt)
    X_notrt = (
        deflate_mtx(Xs, pls_trt.t_[:, 0])
        if ncomp_trt == 1
        else Xs - pls_trt.t_ @ pls_trt.a_.T
    )
    # stage2: batch
    pls_bat = PLSDA(
        ncomp=ncomp_bat, keepX=keepX_bat or [p] * ncomp_bat, tol=tol, max_iter=max_iter
    )
    pls_bat.fit(X_notrt, Ys_bat)
    # deflate all batch components from Xs
    X_temp = Xs.copy()
    for h in range(ncomp_bat):
        X_temp = deflate_mtx(X_temp, pls_bat.t_[:, h])
    # back-transform
    # unscale
    X_nobat = X_temp * np.std(X, axis=0) + np.mean(X, axis=0)
    df_corr = pd.DataFrame(X_nobat, index=df.index, columns=cols)
    df_all = pd.concat([y_sample, y_trt, y_batch, df_corr], axis=1)
    return df_all


# ConQuR - analog to R function and PyPI implementation


class ConQur(TransformerMixin, BaseEstimator):
    """
    Conditional Quantile Regression (ConQuR) batch correction transformer.

    Parameters
    ----------
    batch_cols : list of str
        List of batch column names.
    covariate_cols : list of str
        List of covariate column names.
    reference_batch : dict
        Dictionary specifying reference batch values for each batch column.
    quantiles : tuple of float, optional
        Quantiles to use for quantile regression, by default (0.05, 0.5, 0.95).
    logistic_kwargs : dict, optional
        Keyword arguments for LogisticRegression.
    quantile_kwargs : dict, optional
        Keyword arguments for QuantileRegressor.

    Attributes
    ----------
    _logit_models : dict
        Fitted logistic regression models for zero-mass.
    _quantile_models : dict
        Fitted quantile regression models for nonzero values.
    _col_order : list
        Order of columns used in the model.
    _feature_cols : list
        List of feature columns.
    """

    def __init__(
        self,
        batch_cols,
        covariate_cols,
        reference_batch,  # e.g. {'batch': 0}
        quantiles=(0.05, 0.5, 0.95),
        logistic_kwargs=None,
        quantile_kwargs=None,
    ):
        self.batch_cols = batch_cols
        self.covariate_cols = covariate_cols
        self.reference_batch = reference_batch
        self.quantiles = np.array(quantiles)
        self.logistic_kwargs = logistic_kwargs or {}
        self.quantile_kwargs = quantile_kwargs or {}
        self._logit_models = {}
        self._quantile_models = {}
        self._col_order = []
        self._feature_cols = []

    def fit(self, df, y=None):
        df = df.copy()
        # encode batch/covariates
        for col in self.batch_cols + self.covariate_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.Categorical(df[col]).codes

        # identify features
        reserved = set(self.batch_cols + self.covariate_cols)
        numeric = df.select_dtypes(include="number").columns.tolist()
        self._feature_cols = [c for c in numeric if c not in reserved]
        # define column order in arrays
        self._col_order = self.batch_cols + self.covariate_cols + self._feature_cols

        X_full = df[self._col_order].values.astype(float)
        # build reference array
        X_ref = X_full.copy()
        for col, ref in self.reference_batch.items():
            idx = self._col_order.index(col)
            X_ref[:, idx] = ref

        # design matrices
        bc = [self._col_order.index(c) for c in self.batch_cols]
        cc = [self._col_order.index(c) for c in self.covariate_cols]
        design_idx = bc + cc
        Xd = X_full[:, design_idx]
        # Xd_ref = X_ref[:, design_idx]

        n, p = X_full.shape
        feat_idx = list(range(len(design_idx), p))

        # fit per‐feature models
        for f in feat_idx:
            yv = X_full[:, f]
            ybin = (yv != 0).astype(int)

            # logistic
            if ybin.sum() == 0:
                self._logit_models[f] = ("all_zero", None)
            elif ybin.sum() == n:
                self._logit_models[f] = ("all_one", None)
            else:
                lr = LogisticRegression(**self.logistic_kwargs)
                lr.fit(Xd, ybin)
                self._logit_models[f] = ("model", lr)

            # quantile models on nonzero
            mask = yv != 0
            Xnz = Xd[mask]
            ynz = yv[mask]
            qmods = {
                q: QuantileRegressor(quantile=q, **self.quantile_kwargs).fit(Xnz, ynz)
                for q in self.quantiles
            }
            self._quantile_models[f] = qmods

        return self

    def transform(self, df):
        df_base = df
        df = df.copy()
        # encode again
        for col in self.batch_cols + self.covariate_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.Categorical(df[col]).codes

        X_full = df[self._col_order].values.astype(float)
        X_ref = X_full.copy()
        for col, ref in self.reference_batch.items():
            idx = self._col_order.index(col)
            X_ref[:, idx] = ref

        bc = [self._col_order.index(c) for c in self.batch_cols]
        cc = [self._col_order.index(c) for c in self.covariate_cols]
        design_idx = bc + cc
        Xd = X_full[:, design_idx]
        Xd_ref = X_ref[:, design_idx]

        n, p = X_full.shape
        feat_idx = list(range(len(design_idx), p))
        Xcorr = X_full.copy()

        # apply mapping
        for f in feat_idx:
            yv = X_full[:, f]
            tag, lm = self._logit_models[f]

            # zero‐mass probabilities
            if tag == "all_zero":
                p0 = np.ones(n)
                p0_ref = np.ones(n)
            elif tag == "all_one":
                p0 = np.zeros(n)
                p0_ref = np.zeros(n)
            else:
                p0 = lm.predict_proba(Xd)[:, 0]
                p0_ref = lm.predict_proba(Xd_ref)[:, 0]

            # nonzero quantile predictions
            qmods = self._quantile_models[f]
            Q = np.vstack([qmods[q].predict(Xd) for q in self.quantiles]).T
            Q_ref = np.vstack([qmods[q].predict(Xd_ref) for q in self.quantiles]).T

            # for each sample compute corrected value
            new = np.zeros_like(yv, dtype=float)
            for i in range(n):
                if yv[i] == 0:
                    # keep zero if p0_ref high, else use smallest nonzero quantile
                    new[i] = 0.0 if p0_ref[i] >= 0.5 else Q_ref[i, 0]
                else:
                    # find conditional quantile rank q
                    # solve F = p0 + (1−p0)*q  ⇒  q = (F−p0)/(1−p0)
                    # approximate F by finding which quantile bin y falls into:
                    diffs = np.abs(Q[i] - yv[i])
                    bin_idx = np.argmin(diffs)
                    qbin = self.quantiles[bin_idx]
                    # overall CDF position
                    F = p0[i] + (1 - p0[i]) * qbin
                    # invert in reference: find idx where p0_ref + (1−p0_ref)*quantiles ≈ F
                    # solve q_ref = (F−p0_ref)/(1−p0_ref)
                    q_ref = (
                        (F - p0_ref[i]) / (1 - p0_ref[i])
                        if (1 - p0_ref[i]) > 0
                        else 0.0
                    )
                    # clamp q_ref to [0,1]
                    q_ref = min(max(q_ref, 0.0), 1.0)
                    # choose closest available quantile
                    idx_ref = np.argmin(np.abs(self.quantiles - q_ref))
                    new[i] = Q_ref[i, idx_ref]
            Xcorr[:, f] = new

        # Clamp negatives to zero
        Xcorr[:, len(design_idx) :] = np.where(
            Xcorr[:, len(design_idx) :] < 0, 0, Xcorr[:, len(design_idx) :]
        )

        # rebuild DataFrame
        df_out = pd.DataFrame(
            Xcorr.astype(int), index=df.index, columns=self._col_order
        )
        # Drop non-counts columns
        df_out = df_out.drop([c for c in self.batch_cols], axis=1)
        df_out = df_out.drop([c for c in self.covariate_cols], axis=1)
        df_out = df_out.drop(
            [c for c in df.columns if c not in df_base.columns], axis=1
        )
        other = [c for c in df.columns if c not in self._col_order]
        return pd.concat(
            [
                df_base[other],
                df_base[self.batch_cols],
                df_base[self.covariate_cols],
                df_out,
            ],
            axis=1,
        )


def correctConQuR(
    df,
    batch_cols,
    covariate_cols,
    reference_batch=None,
    quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
    logistic_kwargs={"penalty": "l2", "solver": "lbfgs", "max_iter": 200},
    quantile_kwargs={"alpha": 0.0},
):
    """
    Conditional logistic quantile regression (ConQuR) for batch correction.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing OTU counts and metadata.
    batch_cols : list of str
        List of batch column names.
    covariate_cols : list of str
        List of covariate column names.
    reference_batch : dict, optional
        Dictionary specifying reference batch values for each batch column.
        If None, uses zeros for all batch columns.
    quantiles : tuple of float, optional
        Quantiles to use for quantile regression, by default (0.05, 0.25, 0.5, 0.75, 0.95).
    logistic_kwargs : dict, optional
        Keyword arguments for LogisticRegression.
    quantile_kwargs : dict, optional
        Keyword arguments for QuantileRegressor.

    Returns
    -------
    pandas.DataFrame
        Batch-corrected DataFrame.
    """
    # Define reference batch
    if reference_batch is None:
        reference_batch = {
            batch: ref
            for batch, ref in zip(
                batch_cols, np.zeros(len(batch_cols), dtype=int), strict=False
            )
        }
    # Create model
    conq = ConQur(
        batch_cols=batch_cols,
        covariate_cols=covariate_cols,
        reference_batch=reference_batch,
        quantiles=quantiles,
        logistic_kwargs=logistic_kwargs,
        quantile_kwargs=quantile_kwargs,
    )
    # Fit data into model
    conq.fit(df)
    # Correction
    df_corrected = conq.transform(df)

    return df_corrected


# ComBat-seq
def correctCombatSeq(
    data,
    sample_label,
    batch_label,
    condition_label,
    ref_batch=None,
):
    """
    Perform ComBat-seq batch correction for count data.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing count data and metadata.
    sample_label : str
        Column name for sample identifiers.
    batch_label : str
        Column name for batch identifiers.
    condition_label : str
        Column name for condition/experiment identifiers.
    ref_batch : str or None, optional
        Reference batch to use, by default None.

    Returns
    -------
    pandas.DataFrame
        DataFrame with sample, batch, condition, and ComBat-seq corrected counts.
    """
    count_data = data.select_dtypes(include="number")
    batch_data = [batch for batch in data[batch_label]]
    cov_data = [exp for exp in data[condition_label]]

    corrected = pycombat_seq(
        count_data.T, batch_data, covar_mod=cov_data, ref_batch=ref_batch
    )
    corrected_df = pd.DataFrame(
        corrected.T, index=data.index, columns=count_data.columns
    )

    return pd.concat(
        [data[sample_label], data[batch_label], data[condition_label], corrected_df],
        axis=1,
    )
