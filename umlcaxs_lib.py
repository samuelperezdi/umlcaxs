'''
Unsupervised Machine Learning for the Classification of Astrophysical X-ray Sources (UMLCAXS)

Víctor Samuel Pérez-Díaz, Rafael Martínez-Galarza, Alexander Caicedo-Dorado, Raffaele D'Abrusco

This is the UMLCAXS library. All important functions and procedures that are used in our analysis are stored here.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    '''
    If log is true
    apply log transform adding the minimum non-zero value divided by ten in order to preserve zero properties, then normalize,
    else just normalize.
    '''
    X = X_df.copy(deep=True).to_numpy()
    for name_desc in features:
        col = X_df.columns.get_loc(name_desc)
        X_desc = X_df[name_desc]
        
        if name_desc in features_lognorm:
            nonzero = X_desc[X_desc!=0]
            minval = np.min(nonzero)/10

            X_desc = X_desc + minval

            x = np.log(X_desc.values)
        elif name_desc in features_norm:
            x = X_desc.to_numpy()
        else:
            continue
        min_max_scaler = MinMaxScaler(feature_range=(0,1))
        x_scaled = min_max_scaler.fit_transform(x.reshape(-1,1))
        X[:,col] = x_scaled.flatten()

    X_df_return = pd.DataFrame(X, columns=X_df.columns).astype(X_df.dtypes.to_dict())
    return X_df_return

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

def mahalanobis(x=None, data=None, pseudo=False):
    '''Compute the Mahalanobis Distance between each row of x and the distribution of data. 
    Adapted from https://www.machinelearningplus.com/statistics/mahalanobis-distance/.
    x    : dataframe with observations.
    data : dataframe of the distribution from which the distance is to be computed.
    '''
    data = data.astype(float)
    x = x.astype(float)
    x_mean_subs = x - np.mean(data, axis=0)
    
    cov = np.cov(data.values.T)

    invcov = pinv(cov)
    t_one = np.dot(x_mean_subs, invcov)
    t_two = x_mean_subs.T
    mahal = np.dot(t_one, t_two)
    
    return mahal.diagonal()
    
def create_summary_tables(df):
    '''
    Creates a summary table of the number of source detection by each class in a dataframe of the CSCq + SIMBAD dataset. 
    '''
    data_n = df.copy(deep=True)
    count_obs = data_n.groupby(['main_type']).size()
    df_n = pd.concat([count_obs], axis=1)
    df_n = df_n.rename(columns={0:'size'})
    return df_n.sort_values(by='size', ascending=False)

def softmin(x):
    '''
    Softmin function.
    '''
    return np.exp(-np.abs(x))/sum(np.exp(-np.abs(x)))

def mahal_classifier_cl(cl, cl_raw, features, ltypes, uks=[], singular_out_mode=True):
    '''
    Classify source detections using the Mahalanobis Distance for a particular cluster.
    cl: dataframe of the cluster.
    features: list of feature names to consider.
    ltypes: list of classes names to consider.
    uks: list of classes names that are ambiguous.
    singular_out_mode: boolean determining if the method forces 0 probability for the classes 
    with a singular matrix (number of observations < number of features), or if it uses the
    pseudoinverse instead.
    '''
    if uks:
        cl_nan = cl[(cl.main_type == 'NaN') | cl.main_type.isin(uks)]
        cl_nan_raw = cl_raw[(cl_raw.main_type == 'NaN') | cl_raw.main_type.isin(uks)]
    else:
        cl_nan = cl[cl.main_type == 'NaN']
        cl_nan_raw = cl_raw[cl_raw.main_type == 'NaN']

    cl_nan_feat = cl_nan[features]
    ltypes_distances = []
    for t in ltypes:
        cl_type = cl[cl.main_type == t]
        cl_type_feat = cl_type[features]
        if cl_type_feat.shape[0] < cl_type_feat.shape[1] and singular_out_mode:
            ltypes_distances.append([np.inf]*cl_nan_feat.shape[0])
            continue

        o_mahal_distance = mahalanobis(cl_nan_feat, cl_type_feat)
        ltypes_distances.append(o_mahal_distance)
    
    ltypes_dis_np = np.column_stack(ltypes_distances)

    sm_probs = np.apply_along_axis(softmin, 1, ltypes_dis_np)
    t_amax = np.apply_along_axis(np.argmax, 1, sm_probs)
    types_comp = [ltypes[tname] for tname in t_amax]
    types_probs = pd.DataFrame(sm_probs, columns=ltypes, index=cl_nan.index)

    firstcols = ['name', 'obsid', 'cluster', 'main_type']
    diffcols = [i for i in cl_nan.columns if i not in firstcols]
    out_classification = cl_nan[firstcols].join(types_probs)
    out_classification['main_type'] = types_comp
    out_classification = pd.concat([out_classification.reset_index(drop=True), cl_nan_raw[diffcols].reset_index(drop=True)], axis=1)

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
    '''
    Wrapper of method mahal_classifier_cl, takes a complete dataframe and classify it using our
    cluster-wise classification technique.
    '''
    classified_cls= []
    for cl_i in range(6):
        print(f'***Cluster {cl_i}***')
        data_cl = data[data.cluster == cl_i]
        data_cl_raw = data_raw[data_raw.cluster == cl_i]
        cl_class = mahal_classifier_cl(data_cl, data_cl_raw, features, ltypes, uks, singular_out_mode)
        classified_cls.append(cl_class)
        
    classified_df = pd.concat(classified_cls)
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