# TensorFlow 2 Migration

We're migrating all SNN training components from TensorFlow 1 to TensorFlow 2.

## Status

_Last updated: 06-Oct-20_

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
- [X] Rate regularization
- [X] Voltage regularization (new, from Maass group)
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

## Version 1 Layout

```
Repo
│  
├───adex
│   ├───dynamic_rnn_with_gradients.py
│   │       adex_toy_network.py
│   │       adex_toy_problem.py
│   │       dynamic_rnn_with_gradients()
│   │       list_to_namedtuple()
│   │       namedtuple_to_list()
│   │       New Text Document.txt
│   │       tf_get_shape()
│   │       
│   ├───models.py
│   │   │   compute_fast_gradients()
│   │   │   exp_convolve()
│   │   │   get_signed_weights_clipped()
│   │   │   get_signed_weights_reflect()
│   │   │   pseudo_derivative()
│   │   │   spike_encode()
│   │   │   spike_function()
│   │   │   tile_matrix_to_match()
│   │   │   
│   │   ├───AdEx <class>
│   │   │       compute_z()
│   │   │       output_size()  # property
│   │   │       set_weights()
│   │   │       state_size()   # property
│   │   │       zero_state()
│   │   │       __call__()
│   │   │       __init__()
│   │   │       
│   │   ├───AdExGerstner <class>
│   │   │       compute_z()
│   │   │       output_size()  # property
│   │   │       state_size()   # property
│   │   │       zero_state()
│   │   │       __call__()
│   │   │       __init__()
│   │   │       
│   │   ├───AdEx_Singular <class>
│   │   │       compute_z()
│   │   │       output_size()  # property
│   │   │       set_weights()
│   │   │       state_size()   # property
│   │   │       zero_state()
│   │   │       __call__()
│   │   │       __init__()
│   │   │       
│   │   ├───CustomALIF  <class>
│   │   │       compute_z()
│   │   │       output_size()  # property
│   │   │       set_weights()
│   │   │       state_size()   # property
│   │   │       zero_state()
│   │   │       __call__()
│   │   │       __init__()
│   │   │       
│   │   └───LIF <class>
│   │           output_size()  # property
│   │           set_weights()
│   │           state_size()   # property
│   │           zero_state()
│   │           __call__()
│   │           __init__()
│   │           
│   └───toolbox
│       ├───file_saver_dumer_no_h5py.py
│       │   │   compute_or_load()
│       │   │   flag_to_dict()
│       │   │   get_storage_path_reference()
│       │   │   load_file()
│       │   │   save_file()
│       │   │   
│       │   └───NumpyAwareEncoder <class>
│       │           default()
│       │           
│       ├───matplotlib_extension.py
│       │       arrow_trajectory()
│       │       hide_bottom_axis()
│       │       raster_plot()
│       │       strip_right_top_axis()
│       │       
│       ├───rewiring_tools.py
│       │       assert_connection_number()
│       │       balance_matrix_per_neuron()
│       │       compute_gradients_with_rewiring_variables()
│       │       get_global_connectivity_bound_assertion()
│       │       max_eigen_value_on_unit_circle()
│       │       random_sparse_signed_matrix()
│       │       rewiring()
│       │       rewiring_optimizer_wrapper()
│       │       sample_matrix_specific_reconnection_number_for_global_fixed_connectivity()
│       │       test_random_sparse_signed_matrix()
│       │       weight_sampler()
│       │       
│       ├───spike_encode.py
│       │       spike_encode()
│       │       
│       ├───tensorflow_einsums
│       │   │   test_bij_jk_to_bik.py
│       │   │   test_bij_ki_to_bkj.py
│       │   │   test_bi_bij_to_bj.py
│       │   │   test_bi_ijk_to_bjk.py
│       │   │   
│       │   └───einsum_re_written.py
│       │           einsum_bij_jk_to_bik()
│       │           einsum_bij_ki_to_bkj()
│       │           einsum_bi_bijk_to_bjk()
│       │           einsum_bi_bij_to_bj()
│       │           einsum_bi_ijk_to_bjk()
│       │           
│       └───tensorflow_utils.py
│               boolean_count()
│               discounted_return()
│               exp_convolve()
│               moving_sum()
│               reduce_variance()
│               tf_discounted_reward_test()
│               tf_downsample()
│               tf_downsample_test()
│               tf_exp_convolve_test()
│               tf_feeding_dict_of_placeholder_tuple()
│               tf_moving_sum_test()
│               tf_repeat()
│               tf_repeat_test()
│               tf_roll()
│               tf_tuple_of_placeholder()
│               variable_summaries()
│               
├───rewiring_tools.py
│       assert_connection_number()
│       balance_matrix_per_neuron()
│       compute_gradients_with_rewiring_variables()
│       get_global_connectivity_bound_assertion()
│       max_eigen_value_on_unit_circle()
│       random_sparse_signed_matrix()
│       rewiring()
│       rewiring_optimizer_wrapper()
│       sample_matrix_specific_reconnection_number_for_global_fixed_connectivity()
│       test_random_sparse_signed_matrix()
│       weight_sampler()
│       
└───tools.py
    │   generate_poisson_noise_np()
    │   raster_plot()
    │   strip_right_top_axis()
    │   
    └───NumpyAwareEncoder <class>
            default()
```
