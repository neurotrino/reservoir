# TensorFlow 2 Migration

We're migrating all SNN training components from TensorFlow 1 to TensorFlow 2.

## Status

_Last updated: 15-Sep-20_

- [x] LIF
- [x] BPTT
- [ ] E & I units in LIF
- [ ] E & I units w/ differing time constants
- [ ] ALIF
- [X] Sparse connectivity probabilities
- [ ] Rewiring during training
- [ ] eProp1 symmetric feedback
- [ ] eProp1 random feedback
- [X] AdEx
- [X] E & I units in AdEx
- [ ] Rate regularization
- [ ] Voltage regularization (new, from Maass group)
- [ ] Real-time assessment (plot rasters, print losses to terminal, etc)

## Working on the Project

Begin by reading and understanding `main.py` and `models.py`. Use the 
[Tasklist](https://rb.gy/zuscx6) to check for and claim available tasks.

Don't alter the original (TensorFlow 1) files. If you find bugs therein, make
note and amend them in your TensorFlow 2 version of the file.

## Links

- [Google Doc](rb.gy/gpcgz4)
- [Tasklist](https://rb.gy/zuscx6)

## Executing Scripts

1. Connect to the lab computer (see doc for password)
    ```bash
    ssh macleanlab@205.208.22.225
    ```
2. Enter the TensorFlow 2 virtual environment
    ```bash
    conda activate tf2gpu
    ```
3. You can now execute TensorFlow 2 scripts

## Miscellaneous
- So far we've found `scp` to be the most convenient means of transferring data 
  files
- Use [`screen`](https://linuxize.com/post/how-to-use-linux-screen/) to ensure
  your script keeps running on the server in case your laptop shuts off or 
  explodes
