# Preprocessing
The list of variables:

|NSFW Data                            |AWARE (Article)     |AWARE (Code)             |MASCOTS (Paper)|This Repo (ELEKO 2025)  |
|-------------------------------------|--------------------|-------------------------|---------------|------------------------|
|replica                              |# of replicas       |num_containers           |$n_t$          |num_containers          |
|expected_tps                         |arrival rate        |arrival_rate             |$l_t$          |arrival_rate            |        
|cpu_usage ($\times$ 1000 milicpu)    |n/a                 |avg_cpu_usage            |$\hat{c}_t$    |avg_cpu_usage           |
|cpu ($\times$ 100 milicpu)           |n/a                 |cpu_shares_per_container |$c_t$          |cpu_shares_per_container|
|n/a                                  |cpu utilization     |avg_cpu_util             |$u_t$          |avg_cpu_util            |
|n/a                                  |n/a                 |n/a                      |$\lambda_t$    |arrival_change_rate     |
|num_request                          |throughput          |processing_rate          |$\hat{l}_t$    |processing_rate         |
|n/a                                  |throughtput preservation ratio|data_processing_rate|$\mu_t$   |data_processing_rate    |
|...                                  |...                 |...                      |...            |...                     |


# Experiments
To run the experiments first run preprocessing.py and related experiment file:

~~~bash
# Make sure to run the codes under code folder.
cd code
python preprocessing.py
python experimentX.py
~~~
## Experiment01
No data needed. A very simple dummy test of agent.

|Key|Value|
|---|-----|
|All variables| *num_containers*, *arrival_rate*, *avg_cpu_usage*|
|Action Space (Independent Variables)| *num_containers*: {1,2,3}|
|State| *num_containers*, *arrival_rate*, *avg_cpu_usage*|
|Control Variables|*arrival_rate* = 1|
|Dependent/Measured Variables|*avg_cpu_usage* = *num_containers*/3|
|Reward| *avg_cpu_usage*|
|Data|No data|
|Code| [experiment01.py](code/experiment01.py)|
|What is expected?|After a quick training phase, the agent is expected to take action num_containers=3 unless exploration is used. In the very early steps, you should observe that agent also takes other actions (1 or 2). But after a few iterations, the agent should always choose 3.|


## Experiment02
Based on real-world NSFW data. We first filter [nsfw_raw.csv](data/nsfw_raw.csv) to create a simple subset at [nsfw_experiment2.csv](data/nsfw_experiment2.csv).


|Key|Value|
|---|-----|
|All variables| *num_containers*, *arrival_rate*, *avg_cpu_usage*|
|Action Space (Independent Variables)| *num_containers*: {1,2,3}|
|State| *num_containers*, *arrival_rate*, *avg_cpu_usage*|
|Control Variables|*arrival_rate* = 2|
|Dependent/Measured Variables|*avg_cpu_usage*: random sampling from data filtered based on *num_containers* and *arrival_rate*.|
|Reward| *avg_cpu_usage*|
|Data| [nsfw_experiment2.csv](data/nsfw_experiment2.csv)|
|Preprocessing| Check [preprocessing.py](code/preprocessing.py)
|Code| [experiment02.py](code/experiment02.py)|
|What is expected?|Unlike the first experiment, the agent should converge to action=1 since there is a reverse correlation between *num_containers* and *cpu_avg_usage* and our reward is to maximize *cpu_avg_usage*.|




