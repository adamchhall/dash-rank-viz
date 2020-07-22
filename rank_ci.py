# Data manipulation dependencies
import pandas as pd
import numpy as np

# Specify data location
data_path = 'transit_time.csv'

# Import and clean data
df = pd.read_csv(data_path, skiprows=1)
df = df[['area', 'est_total', 'moe_total']]
df = df.sort_values(by='est_total', ascending=False)
df['rank'] = list(range(1, len(df)+1))

# Add in Bonferroni-corrected joint confidence intervals
# By default MOE is for alpha = 0.1 (z = 1.645).
# Applying Bonferroni correction for 51 states,
# we want alpha = 0.1/51 which corresponds to z = 3.1.
df['total_lb'] = df['est_total'] - (3.1/1.645)*df['moe_total']
df['total_ub'] = df['est_total'] + (3.1/1.645)*df['moe_total']
df['rank_lb']=None
df['rank_ub']=None

# Set debug status
ci_debug = False

# Find joint confidence region for ranks
for area in df.area:
    # Store area confidence interval
    area_k_ci = (float(df[df.area == area]['total_lb']), 
                 float(df[df.area == area]['total_ub']))

    # Find the length of LambdaL_k and LambdaR_k
    LambdaL_k_len = (area_k_ci[1] < df['total_lb']).sum()
    LambdaR_k_len = (area_k_ci[0] > df['total_ub']).sum()

    # Find the overlap and length of LambdaO_k
    overlap = np.maximum(0, np.minimum(area_k_ci[1], df['total_ub']) - np.maximum(area_k_ci[0], df['total_lb']))
    LambdaO_k_len = (overlap!=0).sum()-1

    # Debug Output
    if ci_debug==True:
        print('Area: ', area, '\n')
        print('Theta CI: ', area_k_ci, '\n')
        print('Rank: ', int(df[df.area == area]['rank']), '\n')
        print('Rank CI: ', (LambdaL_k_len + 1, LambdaL_k_len + LambdaO_k_len + 1))
        print('|LambdaL_k|: ', LambdaL_k_len, '\n')
        print('|LambdaR_k|: ', LambdaR_k_len, '\n')
        print('|LambdaO_k|: ', LambdaO_k_len, '\n')

    # Add rank intervals to df
    df.loc[df.area==area, ['rank_lb']] = LambdaL_k_len + 1
    df.loc[df.area==area, ['rank_ub']] = LambdaL_k_len + LambdaO_k_len + 1

    # Reset index
    df = df.reset_index(drop=True)


df['region'] = None

# Define regions

# Northeast
df.loc[df.area.isin(['Connecticut', 'Maine', 
'Massachusetts', 'New Hampshire', 'Rhode Island', 'Vermont', 'New Jersey',
'New York', 'Pennsylvania']), 'region'] = 'Northeast'

# Midwest
df.loc[df.area.isin(['Indiana', 'Illinois', 'Michigan',
'Ohio', 'Wisconsin', 'Iowa', 'Kansas', 'Minnesota', 'Missouri', 'Nebraska',
'North Dakota', 'South Dakota']), 'region'] = 'Midwest'

# West
df.loc[df.area.isin(['California', 'Washington', 'Arizona', 'Colorado',
'Oregon', 'Utah', 'Nevada', 'New Mexico', 'Idaho', 'Montana', 'Wyoming', 'Alaska',
'Hawaii']), 'region'] = 'West'

# South
df.loc[df.area.isin(['Delaware', 'District of Columbia',
'Florida', 'Georgia', 'Maryland', 'North Carolina', 'South Carolina', 'Virginia',
'West Virginia', 'Alabama', 'Kentucky', 'Mississippi', 'Tennessee', 'Arkansas',
'Louisiana', 'Oklahoma', 'Texas']), 'region'] = 'South'

df.to_csv('cleaned_transit_time.csv')