# Configuration Files
Parameters are set within [HJSON](https://hjson.github.io/) files.

## Translation
Most things become namespaces. The `cfg` variable seen in the code is more or
less a 1:1 translation of this HJSON file, with a few convenience features,
like subdirectories un-nesting themselves (e.g. `cfg['save'].checkpoint_dir`).

## Keywords
The format and parser (see `\utils\config.py`) are moderately flexible, but
the following overall structure needs to be maintained:

```JSON
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
            /* this value must be defined */
            checkpoint_dir: some_string

            /* this value must be defined */
            summary_dir: some_string

            /* this value must be defined */
            tb_logdir: some_string

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
