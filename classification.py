from umlcaxs_lib import lognorm, mahal_classifier_all
import pandas as pd

# User-defined constants
FEATURES = ['hard_hm', 'hard_hs', 'hard_ms', 'powlaw_gamma', 'bb_kt',
            'var_prob_b', 'var_ratio_b', 'var_prob_h', 'var_ratio_h',
            'var_prob_s', 'var_ratio_s', 'var_newq_b']
FEATURES_LOGNORM = ['bb_kt', 'var_ratio_h', 'var_ratio_b', 'var_ratio_s', 'var_newq_b']
FEATURES_NORM = ['powlaw_gamma']
LTYPES = ['QSO', 'AGN', 'Seyfert_1', 'Seyfert_2', 'HMXB', 'LMXB', 'XB', 'YSO', 'TTau*', 'Orion_V*']
UKS = ['Star', 'X', 'Radio', 'IR', 'Blue', 'UV', 'gamma', 'PartofG', '**']

# Load data
df_csc_simbad = pd.read_csv('out_data/cluster_csc_simbad.csv', index_col=0)
df_csc_simbad.fillna({'main_type': 'NaN'}, inplace=True)

# Preprocess data
df_csc_out = df_csc_simbad.dropna(subset=FEATURES)
df_csc_lognorm = lognorm(df_csc_out, FEATURES, FEATURES_NORM, FEATURES_LOGNORM)

# Classification
classified_df = mahal_classifier_all(df_csc_lognorm, df_csc_out, FEATURES, LTYPES, uks=UKS)
classified_df.head(10)

# Save results
classified_df.to_csv('./out_data/detection_level_classification.csv')
print('Detections classified.')
