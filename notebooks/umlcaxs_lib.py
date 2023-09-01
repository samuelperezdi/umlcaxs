'''
Unsupervised Machine Learning for the Classification of Astrophysical X-ray Sources (UMLCAXS)

Víctor Samuel Pérez-Díaz, Rafael Martínez-Galarza, Alexander Caicedo-Dorado, Raffaele D'Abrusco

This is the UMLCAXS library. Important functions and procedures that are used in our analysis are stored here.
'''
import pandas as pd
import numpy as np
from astropy.io.votable import parse
from astropy.table import Table
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import pinv


def votable_to_pandas(votable_file):
    '''
    Converts votable to pandas dataframe.
    '''
    votable = parse(votable_file)
    table = votable.get_first_table().to_table(use_names_over_ids=True)
    return table.to_pandas()

def lognorm(X_df, features, features_norm, features_lognorm):
    """
    Normalize and log-transform features based on the given sets of feature names.
    
    Parameters:
        X_df: pandas DataFrame containing the features
        features: List of feature names to be processed
        features_norm: List of feature names to be normalized
        features_lognorm: List of feature names to be log-transformed then normalized
    
    Returns:
        A pandas DataFrame containing the processed features.
    """
    X = X_df.copy(deep=True)
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    
    for feature in features:
        if feature in X.columns:
            column_data = X[feature]
            
            if feature in features_lognorm:
                min_val = np.min(column_data[column_data != 0]) / 10
                transformed_data = np.log(column_data + min_val)
            elif feature in features_norm:
                transformed_data = column_data
            else:
                continue

            scaled_data = min_max_scaler.fit_transform(transformed_data.values.reshape(-1, 1))
            X[feature] = scaled_data.flatten()

    return X.astype(X_df.dtypes.to_dict())

def compute_bics(X, max_clusters, iterations=1):
    '''
    Computes Bayesian Information Criterion for different iterations of Gaussian Mixtures based on a number of components k.
    '''
    bics=[]
    K = range(1, max_clusters)
    
    for k in K:
        # Building and fitting the model
        for i in range(iterations):
            gmm_model = GaussianMixture(n_components=k, covariance_type = 'full').fit(X)
            gmm_model.fit(X)

            bics.append((k, i, gmm_model.bic(X)))
    
    bics_df = pd.DataFrame(bics, columns=['k', 'i', 'bic'])

    return bics_df

def compute_silhouettes(X, max_clusters, iterations=1):
    '''
    Computes Silhouette Scores for different iterations of Gaussian Mixtures based on a number of components k.
    '''
    silhouette_scores=[]
    K = range(2, max_clusters+1)
    
    for k in K:
        # Building and fitting the model
        for i in range(iterations):
            gmm_model = GaussianMixture(n_components=k, covariance_type = 'full').fit(X)
            gmm_model.fit(X)
            
            labels = gmm_model.predict(X)
            sil=silhouette_score(X, labels, metric='euclidean')

            silhouette_scores.append((k, i, sil))
    
    silhouette_scores_df = pd.DataFrame(silhouette_scores, columns=['k', 'i', 'silhouette'])

    return silhouette_scores_df

def mahalanobis(x, data):
    """
    Compute the Mahalanobis Distance between each row of x and the data distribution.
    Adapted from https://www.machinelearningplus.com/statistics/mahalanobis-distance/.

    Parameters:
    x : DataFrame with observations.
    data : DataFrame of the distribution to compute distance from.

    Returns:
    A numpy array containing the Mahalanobis distances.
    """
    x, data = x.astype(float), data.astype(float)
    x_mean_subs = x - np.mean(data, axis=0)
    cov_matrix = np.cov(data.values.T)
    inv_cov_matrix = pinv(cov_matrix)
    
    temp = np.dot(x_mean_subs, inv_cov_matrix)
    mahal_dist = np.dot(temp, x_mean_subs.T).diagonal()
    
    return mahal_dist
    
def create_summary_tables(df):
    """
    Create a summary table of the number of source detections by each class
    in a DataFrame of the cluster CSC + SIMBAD dataset.

    Parameters:
    df : DataFrame containing the dataset.

    Returns:
    A sorted DataFrame summarizing the number of source detections by class.
    """
    grouped_data = df.groupby('main_type').size().reset_index(name='size')
    return grouped_data.sort_values(by='size', ascending=False)

def softmin(x):
    """
    Compute the softmin function on an array.
    
    Parameters:
    x : numpy array or list
    
    Returns:
    Softmin values corresponding to the input array.
    """
    exp_neg_abs_x = np.exp(-np.abs(x))
    return exp_neg_abs_x / exp_neg_abs_x.sum()

def mahal_classifier_cl(cl, cl_raw, features, ltypes, uks=[], singular_out_mode=True):
    """
    Classify source detections using Mahalanobis Distance for a specific cluster.
    
    Parameters:
    cl : DataFrame of the cluster
    cl_raw : Raw DataFrame of the cluster
    features : List of feature names
    ltypes : List of class names
    uks : List of ambiguous class names (optional)
    singular_out_mode : Whether to force 0 probability for classes with singular matrices
    
    Returns:
    DataFrame with classification information
    """
    # Filter data based on 'main_type' and 'uks'
    filter_condition = (cl.main_type == 'NaN') | cl.main_type.isin(uks)
    cl_nan = cl[filter_condition]
    
    # Select feature columns
    cl_nan_feat = cl_nan[features]
    
    # Initialize list to store distances
    ltypes_distances = []
    
    for t in ltypes:
        cl_type = cl[cl.main_type == t]
        cl_type_feat = cl_type[features]
        
        # Handle singular matrix case
        if cl_type_feat.shape[0] < cl_type_feat.shape[1] and singular_out_mode:
            ltypes_distances.append([np.inf]*cl_nan_feat.shape[0])
            continue

        # Compute Mahalanobis distance
        o_mahal_distance = mahalanobis(cl_nan_feat, cl_type_feat)
        ltypes_distances.append(o_mahal_distance)
    
    # Stack distances into a NumPy array and compute softmin probabilities
    ltypes_dis_np = np.column_stack(ltypes_distances)
    sm_probs = np.apply_along_axis(softmin, 1, ltypes_dis_np)
    
    # Find the most probable class for each source
    t_amax = np.argmax(sm_probs, axis=1)
    types_comp = [ltypes[idx] for idx in t_amax]
    
    # Construct output DataFrame
    types_probs = pd.DataFrame(sm_probs, columns=ltypes, index=cl_nan.index)
    firstcols = ['name', 'obsid', 'cluster', 'ra', 'dec', 'main_type']
    out_classification = cl_nan[firstcols].join(types_probs)
    out_classification['main_type'] = types_comp
    
    return out_classification

def process_data_for_validation(data, types, uks):
    '''
    Drops all NaNs and ambiguous classes.
    '''
    if uks:
        df = data[(data.main_type != 'NaN') & ~(data.main_type.isin(uks))]
    else:
        df = data[data.main_type != 'NaN']
    df = df.loc[df['main_type'].isin(types)]
    return df

def mahal_classifier_validation(X_train, X_test, features, ltypes, uks=[], singular_out_mode=True):
    '''
    Wrapper of method mahal_classifier_cl, takes dataframes X_train and X_test for validation. Iterates over
    the cluster subsets to provide a full classification.
    '''
    pred_class = []
    for cl_i in range(6):
        print(f'***Cluster {cl_i}***')
        X_train_cl = X_train[X_train.cluster == cl_i]
        X_test_cl = X_test[X_test.cluster == cl_i]
        X_cl_concat = pd.concat([X_train_cl, X_test_cl])
        test_cl_class = mahal_classifier_cl(cl=X_cl_concat, cl_raw=X_cl_concat, features=features, ltypes=ltypes, uks=uks, singular_out_mode=singular_out_mode)
        pred_class.append(test_cl_class)
        
    pred_class_df = pd.concat(pred_class)
    pred_class_df.sort_values(by=['name', 'obsid'], inplace=True)

    return pred_class_df

def mahal_classifier_all(data, data_raw, features, ltypes, uks=[], singular_out_mode=True):
    """
    Classify an entire dataframe using cluster-wise classification.
    
    Parameters:
    data : DataFrame containing the data
    data_raw : Raw DataFrame of the cluster
    features : List of feature names
    ltypes : List of class names
    uks : List of ambiguous class names (optional)
    singular_out_mode : Whether to force 0 probability for classes with singular matrices
    
    Returns:
    DataFrame with classification information
    """
    classified_dfs = []
    
    for cl_i in range(6):
        print(f'***Cluster {cl_i}***')
        
        data_cl = data[data['cluster'] == cl_i]
        data_cl_raw = data_raw[data_raw['cluster'] == cl_i]
        
        cl_classified = mahal_classifier_cl(data_cl, data_cl_raw, features, ltypes, uks, singular_out_mode)
        classified_dfs.append(cl_classified)
    
    classified_df = pd.concat(classified_dfs)
    classified_df.sort_values(by=['name', 'obsid'], inplace=True)
    classified_df.reset_index(drop=True, inplace=True)
    
    return classified_df


def extract_sources_aladin(df, source_type, agg_type=False):
    if agg_type:
        q_res = df[df['agg_master_class'] == source_type]
        print(f'{len(q_res)} source detections found of aggregated class {source_type}')
    else:
        q_res = df[df['master_class'] == source_type]
        print(f'{len(q_res)} source detections found of class {source_type}...')
    q_res_astro = Table.from_pandas(q_res)
    return q_res_astro