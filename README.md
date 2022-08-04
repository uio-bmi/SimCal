# Simulation_Calibration
Codebase for the SimCalibration project, which seeks to employ experimentalal testing of meta-simulations. It evaluates and calibrates Machine Learning pipelines in simulated words with embedded hyperparameters and benchmarks this towards similar representations of pipelines in the real world. The goal of this project is to understand the environments (e.g., Structural Learning (SL) algorithms, Machine Learning (ML) hyperparameters) which contribute to simulated outcomes being able to learn a data distribution effectively.

The example file (check_waml.py) tests the project by running a simplistic example of a linear combination grouth truth model using a Directed Acyclic Graph (DAG), it specifies level 1 (ML hyperparameters) and level 2 (SL hyperparameters) calibrations and benchmarks real and simulated solutions using the Evaluator class, followed by postprocessing and ploting in the Postprocessing class. The examples of analysis setups up situations in which only real world data is used to benchmark ML methods highlighting the error when performing resampling (analysis 1), the difference in Pearson corrrelation coefficents between a real world and a learned world generating from sampling observational data using structural learning (analysis 2), inter-world benchmarking between real and learned worlds (analysis 3), repeating structural learning and inter-world benchmarking 100 times (analysis 4).

The folder structure is as follows: 
1. Utils folder contains support files to assist in performing bootstrapping and visualising the evaluated results

2. Tests folder performs simplistic black-box testing of using input data samples

3. ml_models folder contains files for initializing ML models

4. dg_models folder contains files for setuping up structural learning to generate learned worlds in contrast to the real worlds which are sampled from. While bayesian network learning packages are used to generate learned worlds, DagSim is used to specify the DAG for real worlds (https://github.com/uio-bmi/dagsim).
