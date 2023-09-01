import numpy as np
import pandas as pd
import itertools
from collections import Counter

# Class definitions
ltypes = ['QSO', 'AGN', 'Seyfert_1', 'Seyfert_2', 'HMXB', 'LMXB', 'XB', 'YSO', 'TTau*', 'Orion_V*']
grouped_replace = {'QSO': 'AGN', 'Seyfert_1': 'Seyfert', 'Seyfert_2': 'Seyfert', 'HMXB': 'XB', 'LMXB':'XB', 'TTau*':'YSO', 'Orion_V*': 'YSO'}
classified_df = pd.read_csv('./out_data/detection_level_classification.csv', index_col=0)

# Helper functions
def mean_std_round(x):
    return f"{round(np.mean(x), 3)}±{round(np.std(x), 3)}"

def most_common(x): 
    return Counter(x).most_common(1)[0][0]

# Soft voting
summ_table = classified_df.groupby('name')[ltypes].agg(mean_std_round)
summ_table['detection_count'] = classified_df.groupby(['name']).size()
ra_dec_df = classified_df.groupby('name')[['ra', 'dec']].first()
summ_table = summ_table.join(ra_dec_df)
summ_table_prov = classified_df.groupby('name')[ltypes].agg(['mean', 'std'])
class_mean_names = [list(tup) for tup in itertools.product(ltypes, ['mean'], repeat=1)]
names_comp = summ_table_prov[class_mean_names].idxmax(axis=1).to_list()
master_names = [name[0] for name in names_comp]
summ_table['soft_master_class'] = master_names
summ_table.sort_values('detection_count', ascending=False, inplace=True)

# Hard voting
summ_table_hard = classified_df.groupby('name').size().to_frame('detection_count')
summ_table_hard['hard_master_class'] = classified_df.groupby('name')['main_type'].agg(most_common)
ra_dec_df = classified_df.groupby('name')[['ra', 'dec']].first()
summ_table_hard = summ_table_hard.join(ra_dec_df)
summ_table_hard.sort_values('detection_count', ascending=False, inplace=True)

# Uniquely classified
uniquely_classified_df = summ_table[summ_table.soft_master_class == summ_table_hard.hard_master_class].copy()
uniquely_classified_df = uniquely_classified_df.rename(columns={'soft_master_class':'master_class'})
uniquely_classified_df['agg_master_class'] = uniquely_classified_df['master_class'].replace(grouped_replace)
unique_cols = ['agg_master_class', 'master_class', 'detection_count'] + ltypes + ['ra', 'dec']
uniquely_classified_df_out = uniquely_classified_df[unique_cols]
uniquely_classified_df_out.to_csv('./out_data/uniquely_classified.csv', encoding='utf-8')
print('Uniquely classified table generated.')

# Ambiguous
ambiguous_class_df = summ_table[summ_table.soft_master_class != summ_table_hard.hard_master_class].copy()
ambiguous_class_df['hard_master_class'] = summ_table_hard['hard_master_class']
ambiguous_cols = ['hard_master_class', 'soft_master_class', 'detection_count'] + ltypes + ['ra', 'dec']
ambiguous_class_df_out = ambiguous_class_df[ambiguous_cols]
ambiguous_class_df_out.to_csv('./out_data/ambiguous_classification.csv', encoding='utf-8')
print('Ambiguous table generated.')