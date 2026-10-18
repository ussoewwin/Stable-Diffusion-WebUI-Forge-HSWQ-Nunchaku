# Stub: real comfy_aimdo.malloc_graph provides CUDA graph memory compilation. Forge uses its own loader.

from contextlib import contextmanager


class MallocGraph:
    def __init__(self, *args, **kwargs):
        self._scopes = []

    def push(self, name=None):
        pass

    def pop(self):
        return False

    def abort(self):
        pass

    def pause(self, sync=False):
        pass

    def resume(self, sync=False):
        pass

    @contextmanager
    def use_stream(self, stream):
        yield

    def iterate(self, name=None):
        return False
