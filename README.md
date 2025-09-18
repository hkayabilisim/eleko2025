# Experiment01
No data needed. Based on dummy simulation.
~~~~
cd code
python experiment01.py
~~~~

# Experiment02
Based on real-world NSFW data. 

We first filter raw data to create a simple subset and save under ../data/nsfw_experiment2.csv
~~~~
cd code
python preprocessing.py
~~~~

Then we run our experiment. This time the agent should converge to action=1 since
there is a reverse correlation between num_containers and cpu_avg_usage.
and our reward is to maximize cpu_avg_usage.
~~~~
cd code
python experiment02.py
~~~~
