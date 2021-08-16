# Callbacks (Logging)

Support for `tf.keras.callbacks` is not currently supported, as our models use the subclassing method and requires
finer granularity of logging than these callbacks readily offer. Should `tf.keras.callbacks.Callback` objects be added
to this project, they should be in this directory.
