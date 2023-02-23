# MultiGPU p-tune training with NeMo 

## structure:
 - nemo_model/ : pretrained .nemo model
 - conf/  : yaml file for hyrda automation 
 - data/  : training and testing data files


## How to run:

1. put .nemo pretrained model in nemo_models/

2. modify the slurm script:
   1. --ntasks-per-node=2 # equal to the number of processes per nodes
   2. --gres=gpu:2 # number of GPU on one node. should be equal to --ntasks-per-node
   3. WORLD_SIZE # total number of process occumpied by model or data parallel
   4. CONTAINER # container location
   5. --config-name # select config file

3. edit the corresponding conf/*.yaml for your training purpose
   1. tensor_parallel_size: for tensor or model parallel
   2. language_model_path: where is .nemo file
   3. existing_tasks, new_tasks
   4. train_ds, validation_ds 
   4. resume_if_exists =True/False per user's own goal

There are 3 conf.yaml examples in the conf folder of p-tune/prompt tuning GPT model 