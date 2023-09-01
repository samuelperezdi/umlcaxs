import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from umlcaxs_lib import votable_to_pandas, lognorm

def load_and_preprocess_data(file_path, features, features_lognorm, features_norm):
    data = votable_to_pandas(file_path)
    
    # Define new measurement columns
    data['var_ratio_b'] = data['var_sigma_b'] / data['var_mean_b']
    data['var_ratio_h'] = data['var_sigma_h'] / data['var_mean_h']
    data['var_ratio_s'] = data['var_sigma_s'] / data['var_mean_s']
    data['var_newq_b'] = ((data['var_max_b'] + data['var_min_b']) / 2) / data['var_mean_b']
    
    # Drop data with missing feature values
    data_out = data.dropna(subset=features)
    X_df = data_out[features]
    
    # Normalize or log-normalize features
    return lognorm(X_df, features, features_norm, features_lognorm).to_numpy(), data_out

def cluster_data(X, n_components=6):
    gm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42).fit(X)
    labels = gm.predict(X)
    return labels, gm.predict_proba(X)


# Feature definitions
features = ['hard_hm', 'hard_hs', 'hard_ms', 'powlaw_gamma', 'bb_kt', 'var_prob_b', 'var_ratio_b', 'var_prob_h', 'var_ratio_h', 'var_prob_s', 'var_ratio_s', 'var_newq_b']
features_lognorm = ['bb_kt', 'var_ratio_h', 'var_ratio_b', 'var_ratio_s', 'var_newq_b']
features_norm = ['powlaw_gamma']

X, data_out = load_and_preprocess_data("data/cscresults.vot", features, features_lognorm, features_norm)
labels, probabilities = cluster_data(X)

# Add cluster labels to original data and save to CSV
data_out['cluster'] = labels
data_out.to_csv('out_data/cluster_csc.csv')

# Save cluster probabilities
df_provs = pd.DataFrame(probabilities, columns=range(6))
df_provs['cluster'] = labels
# Use df_provs for further analysis if needed

print('Clustering finished.')
