#%%
import pandas as pd

############################
# Raw Data
############################
df = pd.read_csv('../data/nsfw_raw.csv')
df = df.sort_values(by=['timestamp'])
row_count = df.shape[0]
print(f'Importing raw data...{df.shape[0]} rows imported')

# drop rows when cpu_usage is missing or less than or equal to zero
df = df.dropna(subset=['cpu_usage'])
df = df[df['cpu_usage'] > 0]
print(f'Dropped {row_count - df.shape[0]} rows with missing or non-positive cpu_usage')
row_count = df.shape[0]

#############################
# Data prep for Experiment02
#############################
# take subset of df where cpu=40 and expected_tps=2
print(f'Preparing experiment02 data...fixing a specific cpu and expected_tps')
df_experiment02 = df[(df['cpu'] == 40) & (df['expected_tps'] == 2)]


# drop all columns except replica, cpu_usage and expected_tps
df_experiment02 = df_experiment02[['replica','expected_tps','cpu_usage']]
# rename columns to match AWARE
df_experiment02 = df_experiment02.rename(columns={'replica':'num_containers',
                                      'cpu_usage':'avg_cpu_usage',
                                      'expected_tps':'arrival_rate'})
# Data for Experiment2
df_experiment02.to_csv('../data/nsfw_experiment2.csv',index=False)
print('Saved ../data/nsfw_experiment2.csv')


#############################
# Data prep for Experiment03
#############################
# take subset of df where cpu=40
print(f'Preparing experiment03 data...fixing a specific cpu and varying expected_tps')

df_experiment03 = df.copy()
df_experiment03["previous_expected_tps"] = df_experiment03["expected_tps"].shift(fill_value=1)

df_experiment03 = df_experiment03[(df['cpu'] == 40)]

df_experiment03['cpu_utilization'] = (df_experiment03['cpu_usage'] / (df_experiment03['cpu'] / 10.0)).clip(upper=1.0)

# drop all columns except replica, cpu_usage and expected_tps
df_experiment03 = df_experiment03[['replica', 'expected_tps', 'instant_tps', 'cpu_utilization', 'previous_expected_tps', 'response_time']]
# rename replica to num_containers, cpu_usage to avg_cpu_util and expected_tps to arrival_rate
df_experiment03 = df_experiment03.rename(columns={'replica':'num_containers',
                                      'cpu_utilization':'avg_cpu_util',
                                      'expected_tps':'arrival_rate',
                                      'previous_expected_tps':'prev_arrival_rate'})
# Data for Experiment2
df_experiment03.to_csv('../data/nsfw_experiment3.csv',index=False)
print('Saved ../data/nsfw_experiment3.csv')

#############################

# Data prep for Experiment04
#############################
print(f'Preparing experiment03 data with varying cpu and varying expected_tps')

df_experiment04 = df.copy()
df_experiment04["previous_expected_tps"] = df_experiment04["expected_tps"].shift(fill_value=1)

df_experiment04['cpu_utilization'] = (df_experiment04['cpu_usage'] / (df_experiment04['cpu'] / 10.0)).clip(upper=1.0)

# drop all columns except replica, cpu_usage and expected_tps
df_experiment04 = df_experiment04[['replica', 'cpu', 'expected_tps', 'instant_tps', 'cpu_utilization', 'previous_expected_tps', 'response_time']]
# rename replica to num_containers, cpu_usage to avg_cpu_util and expected_tps to arrival_rate
df_experiment04 = df_experiment04.rename(columns={'replica':'num_containers',
                                      'cpu_utilization':'avg_cpu_util',
                                      'expected_tps':'arrival_rate',
                                      'previous_expected_tps':'prev_arrival_rate'})
# Data for Experiment2
df_experiment04.to_csv('../data/nsfw_experiment4.csv',index=False)
print('Saved ../data/nsfw_experiment4.csv')

# Data prep for Experiment05
#############################
def prep_experiment5(df):
    # Copy only index
    df_exp = pd.DataFrame(index=df.index.copy())

    # Copy/transform attributes
    df_exp['num_containers']           = df['replica']
    df_exp['arrival_rate']             = df['expected_tps']
    df_exp['cpu_shares_per_container'] = df['cpu'] * 100            # milicpu
    df_exp['avg_cpu_usage']            = df['cpu_usage'] * 1000     # milicpu
    df_exp['latency']                  = df['response_time'] / 1000 # seconds
    df_exp['processing_rate']          = df['num_request']

    # New attributes
    df_exp['previous_arrival_rate']    = df_exp['arrival_rate'].shift(fill_value=1)
    df_exp['arrival_change_rate']      = df_exp['arrival_rate'] / df_exp['previous_arrival_rate']
    df_exp['cpu_utilization']          = (df_exp['avg_cpu_usage'] / df_exp['cpu_shares_per_container']).clip(upper=1.0)
    df_exp['data_processing_rate']     = df_exp['processing_rate'] / df_exp['arrival_rate']

    df_exp.to_csv('../data/nsfw_experiment5.csv',index=False)
    print('Processing experiment05 and saving into ../data/nsfw_experiment5.csv')

prep_experiment5(df)
