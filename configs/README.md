# Configuration Files
Parameters are set within [HJSON](https://hjson.github.io/) files.

## Translation
Most things become namespaces. The `cfg` variable seen in the code is more or
less a 1:1 translation of this HJSON file, with a few convenience features,
like subdirectories un-nesting themselves (e.g. `cfg['save'].checkpoint_dir`).

## The Six Sections

### Model
Parameters used to instantiate your model. Can contain submodels and must be consistent with a) the variables passed in
the model's `.__init__()` method and b) the `template` variable in your main script.

### Save
Parameters and flags configuring where your data gets saved and how.

### Data
Parameters used to generate or describe your datasets.

### Log
Parameters used to configure TensorFlow logging and your logger objects (should not be used to configure Python's
`logging` module).

**Profiling.** Uses TensorBoard's profiler. Output is dumped into `profile_dir`. Configure with `run_profiler` and `profiler_epochs`.

### Train
Parameters configuring how the model is trained.

### Misc
If it doesn't belong in one of the previous five categories, but would be useful to have in a configuration file, put
it here.

## Reserved Labels
The format and parser (see `\utils\config.py`) are moderately flexible, but
the following overall structure needs to be maintained:

```HJSON
{
    /* only the following six namespaces will be acknowledged */

    model:
    {
        /* add (potentially nested) primitives and/or classes here */
    }

    save:
    {
        /* this value must be defined */
        exp_dir: some_string

        /* this namespace must be defined */
        subdirs:
        {
            /* add additional subdirectories you want created here; use misc
             * for any directories you won't be saving data into */
        }

        /* this value must be defined */
        avoid_overwrite: bool

        /* this value must be defined */
        hard_overwrite: bool

        /* this value must be defined */
        timestamp: bool

        /* this value must be defined */
        log_config: bool

        /* add (potentially nested) primitives here */
    }

    data:
    {
        /* add (potentially nested) primitives here */
    }

    log:
    {
        /* add (potentially nested) primitives here */
    }

    train:
    {
        /* add (potentially nested) primitives here */
    }

    misc:
    {
        /* add (potentially nested) primitives here */
    }
}

```

Additionally, if you want a model or submodel to have access to the `cfg` variable (which is a transcription of the
HJSON file into a six-keyed dictionary of `SimpleNamespace`s), you must add `cfg: null` in its parameters. Because of
this, `cfg` is a reserved variable.
