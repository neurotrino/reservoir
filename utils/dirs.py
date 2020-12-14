import os

def create_dirs(save_cfg):
    """TODO: docs"""
    dirs = [
        save_cfg.summary_dir,     # summary information about training
        save_cfg.checkpoint_dir,  # training checkpoints
        save_cfg.log_dir          # TensorBoard logdirs
    ]
    try:
        for _dir in dirs:
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
