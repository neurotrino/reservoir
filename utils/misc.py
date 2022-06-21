"""Generic wrapper to make decorators switchable."""

class SwitchedDecorator:
    """Wrapper providing a programmatic toggle for decorators."""

    def __init__(self, decorator_fn, enabled=True):
        """Instantiate object."""
        self._enabled = enabled
        self._decorator_fn = decorator_fn

    def __call__(self, target):
        """Runs the function (un)decorated if (not) enabled."""
        if self._enabled:
            return self._decorator_fn(target)
        return target

    @property
    def enabled(self):
        """Flag indicating decorator status."""
        return self._enabled

    @enabled.setter
    def enabled(self, new_val):
        """Setter to update the `enabled` flag."""
        if not isinstance(new_val, bool):
            raise ValueError(
                f"expected boolean flag, got value '{new_val}' of type " +
                f"{type(new_val)}"
            )
        self._enabled = new_val
