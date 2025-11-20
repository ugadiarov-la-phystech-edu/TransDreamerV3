# TransDreamerV3: Implanting Transformer In DreamerV3

# TODO
* Fix policy for Homegrid
* Add videos to comet loging


## About this repo

### TransDreamerV2/ is cloned from [repo](https://github.com/changchencc/TransDreamer/tree/main)

* the .sh files were used to trained models for different tasks
* the notes.txt contains notes on how to setup the environment

### dreamerv3_8/ is cloned from [repo](https://github.com/danijar/dreamerv3/tree/main)

* the .sh files were used to trained models for different tasks
* requirements.txt and requirements_3_8.txt defined packages needed for running the code. I was able to run everything except Minecraft using python 3.8.0 and packages inside requirements_3_8.txt. Remember to install jax first 

```
pip install --upgrade pip

# for cuda 11
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

for cuda 12
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```


### transdreamerv3_8/ contains our model which was built based on [DreamerV3](https://github.com/danijar/dreamerv3/tree/main)

* Same environments is used as with DreamerV3
* the .sh files were used to trained models for different tasks

### results/ contains code for making figures and figures included in the paper. 

Model checkpoints are available upon request. 


## Examples of run


```
COMET_API_KEY=YOUR_API dreamer/train.py \
--logdir ~/dreamers/TDV3_logdir/atari_boxing \  
--configs atari \
--task atari_boxing \
--loss_scales.lm 0  \
--run.use_comet True \
--run.comet_project transdreamerv \
--run.comet_workspace dreamerv3

```

