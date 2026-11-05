"""Process-local bounded handles with a canonical HDF5 fallback."""
from collections import OrderedDict
import os
from pathlib import Path
import warnings
from vpp2.libero.motion_data import load_motion_clip as load_hdf5_clip
from .motion_records import MotionRecords, cache_path, source_signature, normalize_camera_maps
from .views import RGB_CAMERAS, motion_cameras


class MotionReader:
    def __init__(self, backend='auto', max_open=8, normalization_cameras=RGB_CAMERAS):
        if backend not in ('auto', 'hdf5', 'records'):
            raise ValueError('motion_backend must be auto, hdf5, or records')
        if max_open < 1: raise ValueError('max_open must be positive')
        self.backend, self.max_open = backend, int(max_open)
        # Low-level storage tools retain dual-view defaults. Training explicitly
        # passes its supervised views, currently just the external camera.
        self.normalization_cameras = motion_cameras(normalization_cameras)
        self._pid = os.getpid()
        self._open = OrderedDict()
        self._warned = set()
        self.last_backend = None

    def close(self):
        for _, handle in self._open.values(): handle.close()
        self._open.clear()

    def __del__(self):
        if hasattr(self, '_open'):
            self.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_open'] = OrderedDict()  # spawn workers open their own descriptors
        return state

    def _get(self, source):
        # Fork workers must never share inherited HDF5/indexed-reader state.
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        path = cache_path(source)
        signature = source_signature(source)
        st = path.stat()
        key = str(path)
        identity = (st.st_size, st.st_mtime_ns, signature['size'], signature['mtime_ns'])
        if key in self._open:
            previous, handle = self._open.pop(key)
            if previous == identity:
                self._open[key] = (identity, handle)
                return handle
            handle.close()
        handle = MotionRecords(path)
        source = Path(source)
        if (handle.meta.get('source_signature') != signature or handle.meta.get('replay_version') != 3
                or handle.meta.get('source_name') != source.name
                or handle.meta.get('source_dataset') != source.parents[2].name):
            handle.close()
            raise ValueError('Stale fast motion cache')
        self._open[key] = (identity, handle)
        while len(self._open) > self.max_open:
            _, (_, old) = self._open.popitem(last=False)
            old.close()
        return handle

    def load(self, source, *, start=0, length=8, stride=4, flow_dt=4, prediction_horizon=32):
        kwargs = dict(start=start, length=length, stride=stride, flow_dt=flow_dt,
                      prediction_horizon=prediction_horizon)
        compatible = length == 8 and stride == 4 and flow_dt in (1,4) and start >= 0
        if self.backend != 'hdf5' and compatible:
            try:
                handle = self._get(source)
                result = handle.clip(**kwargs, normalization_cameras=self.normalization_cameras)
                self.last_backend = 'records'
                return result
            except (OSError, ValueError, RuntimeError) as error:
                if self.backend == 'records': raise
                # Never drop a sample or return stale/malformed labels. The
                # untouched HDF5 is authoritative while conversion catches up.
                key = str(cache_path(source))
                if key in self._open:
                    _, handle = self._open.pop(key)
                    handle.close()
                if not isinstance(error, FileNotFoundError) and key not in self._warned:
                    warnings.warn(f'Fast motion cache unavailable ({error}); reading GT HDF5: {source}')
                    self._warned.add(key)
        elif self.backend == 'records':
            raise ValueError('Unsupported fast motion sampling settings')
        self.last_backend = 'hdf5'
        result = load_hdf5_clip(source, normalize='none', **kwargs)
        center, scale = normalize_camera_maps(result['cameras'], self.normalization_cameras)
        result.update(center=center, scale=scale, normalized=True,
                      normalization_cameras=list(self.normalization_cameras))
        return result
