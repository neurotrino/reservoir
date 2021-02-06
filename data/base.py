import tensorflow as tf

class BaseDataGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = None

        self._child_name = self.__class__.__name__

    def get(self):
        """Return the dataset."""
        raise NotImplementedError(
            f"{self._child_name} does not support direct access:"
            + f"use {self._child_name}.next() to iterate over your data"
        )

        return None

    def next(self):
        """Yield the next portion of the dataset."""
        raise NotImplementedError(
            f"{self._child_name} has no custom iterator: "
            + f" use {self._child_name}.get() to directly access your data"
        )
        yield None
