## SSH-ing to servers

SSH to `notchpeak` or `nothchpeak2`: 
```
ssh uNID@notchpeak2.chpc.utah.edu
```

You will be prompted to give your uNID password. 

Outside of the university network, you need to use VPN. For how to access SoC VPN, see [here](https://support.cs.utah.edu/index.php/misc/30-pa-vpn-setup#:~:text=Accessing%20the%20School%20of%20Computing's,Active%20Directory%20username%20and%20password).

## Conda environments  

Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers). E.g,. on Linux and python 3.7:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-aarch64.sh
bash Miniconda3-py37_4.12.0-Linux-x86_64.sh
```

Exit and ssh again.

Create and activate a Conda environment with a desirecd python version, e.g., 3.7: 

```
conda create -n <env_name> python=3.7
conda activate <env_name>
```

In a repo with code you're trying to run, you'll usually see a `requirements.txt` file with required packages. If so, you can run: 
```
pip install -r requirements.txt
```

You might get this error: `RuntimeError: CUDA error: no kernel image is available for execution on the device` if your cuda drivers and pytorch version do not match. I personally check cuda drivers by running `nvidia-smi` (upper right corner). You can go to https://pytorch.org/ and get a command for right drivers. E.g., for cuda drivers 11.6 I run: `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116`/

## Cuda drivers 

When SSH-ed on CHPC's servers/nodes, when you run 
`which nvcc` and cuda drivers are loaded, you'll see something like: `/uufs/chpc.utah.edu/sys/spack/linux-rocky8-nehalem/gcc-8.5.0/cuda-11.6.2-hgkn7czv7ciyy3gtpazwk2s72msbw6l2/bin/nvcc`

If not you need to load them: 
* `module spider cuda` which cuda drivers are available through the module list 
* `module load cuda/11.6.2` load cuda 11.6
* `modue list` to see that cuda is loaded 

You need to install pytorch version for your cuda drviers' version, e.g., for 11.6: 
`pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116`. Check https://pytorch.org/ for command you need (search "INSTALL PYTORCH").

## [CHPC](https://www.chpc.utah.edu/documentation/index.php) 

The list of nodes and corresponding GPU types can be found [here](https://www.chpc.utah.edu/documentation/guides/gpus-accelerators.php) under "GPU Hardware Overview > GPU node list". 

SOTA GPUs are A100s. Check the comparison between GPUs [here](https://lambdalabs.com/gpu-benchmarks), and [nvidia](https://www.nvidia.com/en-us/data-center/a100/) says: 
> "A100 provides up to 20X higher performance over the prior generation and can be partitioned into seven GPU instances to dynamically adjust to shifting demands. The A100 80GB debuts the world's fastest memory bandwidth at over 2 terabytes per second (TB/s) to run the largest models and datasets". 

For NLP tasks, regardless of the GPU type, most likely you need at least 24GB of GPU memory.

Note that the nodes with A100's include the two I own (notch369-70), as well as notch 347 and notch348 (these two have 1 GPU each). There are also 40GB A100's on notch330 (quantity 8). Finally on notch293 there are 4 40GB a100s -- this is a general gpu node, so the account and partition would be notchpeak-gpu. You can schedule jobs on any of these, but students of faculty who own these GPUs have a higher priority and sometimes (e.g., in my case) their students jobs preempt other jobs (i.e., other jobs are killed). That's why good checkpointing is important.  

## Queue jobs 
You need to write a batch script, something like this:

```
#!/bin/bash
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8:00:00
#SBATCH --mem=24GB
#SBATCH --mail-user=<your email>
#SBATCH --mail-type=FAIL,END
#SBATCH -o <add some filename>-%j

source ~/miniconda3/etc/profile.d/conda.sh
conda activate <env name>

wandb disabled 
export TRANSFORMER_CACHE="/scratch/general/vast/<your uNID>/huggingface_cache"
python ... 
```
and run: `sbatch <script name>.sh`.

## Interactive job 

For example: 

1. `salloc -A marasovic-gpu-np -p marasovic-gpu-np -n 32 -N 1 --gres=gpu:a100:1 -t 1:00:00 --mem=40GB` 

2. `python ...`

WARNING: Use sporadically. If you allocate a GPU but do not use it, you are wasting the resource! Until your interactive session is not over, other jobs are waiting. 

## Checking jobs 

You can check the queue by running: `squeue` (for everything on CHPC) or `squeue -p marasovic-gpu-np` (for 2 GPUs I owe).

To log into the nodes where the job runs, you need to check the job ID (it will be in the output of the `squeue`), and then run: 

`srun --jobid=jobID --pty /bin/bash -l`.

If you want to see how GPU memory is utilized by running: 

```
ml nvtop 
nvtop
```

## Storage 

Our model checkpoints are huge so we shouldn't save them to our local directories. 

CHPC has scratch file systems (https://www.chpc.utah.edu/resources/storage_services.php) that are helpful. 

Make a directory in one of them: 

`mkdir /scratch/general/vast/<uNID>` 

and save your checkpoints and data there. 

Note however that "On these scratch file system, files that have not been accessed for 60 days are automatically scrubbed." 

## Tmux 

You need to use [tmux](https://github.com/tmux/tmux/wiki) to be able to get back to your experiments once you close your laptop or ssh connection breaks. [Tmux Cheat Sheet & Quick Reference](https://tmuxcheatsheet.com/). 

## Git 

For introduction to Git see [this](https://missing.csail.mit.edu/2020/version-control/). I make changes to my code locally, push them to a repo, pull the changes on server, and then run an experiment on server. 

## Other tips 

### Tip #1
If you use [huggingface](https://huggingface.co/course/chapter1/1), downloaded models will be saved in a `.cache` folder in your home directory. It will quickly eat all of your home directory space, so what I do is: 

```
mkdir /scratch/general/vast/<uNID>/huggingface_cache
export TRANSFORMERS_CACHE="/scratch/general/vast/<uNID>/huggingface_cache"
```

You'll need to do this every time you run a job, so it's good to fix this in the code instead.

### Tip #2

If you use [huggingface](https://huggingface.co/course/chapter1/1), [wandb](https://wandb.ai/site) will be running automatically. However, I constantly run into issues with it so I run: 
`wandb disabled` and not use it.

### Tip #3

Use `import pdb; pdb.set_trace()` or `breakpoint()` in python for debugging. 

### Tip #4 

Use `scancel <jobID>` to cancel your job.