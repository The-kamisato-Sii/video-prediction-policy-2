"""Lossless indexed motion records: one contiguous read per stride-four clip.

Each record holds both cameras' metric float32 geometry, dt1/dt4 flow and
bit-packed validity. Frame records are ordered 0,4,...,1,5,...,2,6,...,3,7,... .
The outer index is VPP-specific; each payload is a standard checksummed LZ4
frame. GT normalization is applied after reading, never baked into the cache.
"""
from __future__ import annotations
import json
import os
import struct
from pathlib import Path
import numpy as np
from vpp2.libero.motion_geometry import canonical_normalization
from .views import motion_cameras

MAGIC = b'VPPMOT01'
SCHEMA = 'vpp2.motion_records.v1'
CAMERAS = ('image', 'wrist_image')


def phase_order(n):
    return np.concatenate([np.arange(r, n, 4, dtype=np.int64) for r in range(4)])


def phase_position(start, n):
    return sum(len(range(r, n, 4)) for r in range(start % 4)) + start // 4


def source_signature(path):
    st = Path(path).stat()
    return dict(size=st.st_size, mtime_ns=st.st_mtime_ns)


def cache_path(source):
    source = Path(source)
    return source.parents[2] / 'motions_fast' / source.parent.name / (source.stem + '.mrec')


def encode_record(values, masks, shuffle=True):
    import lz4.frame
    v = np.ascontiguousarray(values, dtype='<f4')
    if shuffle:
        payload = v.view(np.uint8).reshape(-1, 4).T.copy().tobytes()
    else:
        payload = v.tobytes()
    payload += np.packbits(np.asarray(masks, dtype=np.uint8).reshape(-1), bitorder='little').tobytes()
    return lz4.frame.compress(payload, compression_level=0,
        block_size=lz4.frame.BLOCKSIZE_MAX4MB, content_checksum=True, store_size=True)


def decode_record(blob, height, width, shuffle=True):
    import lz4.frame
    raw = lz4.frame.decompress(blob)
    count = 2 * 3 * height * width * 3
    masks_count = 2 * 3 * height * width
    floats_bytes = count * 4
    if len(raw) != floats_bytes + (masks_count + 7) // 8:
        raise ValueError('Unexpected motion record payload size')
    if shuffle:
        b = np.frombuffer(raw, dtype=np.uint8, count=floats_bytes).reshape(4, count).T.copy()
        values = b.reshape(-1).view('<f4').reshape(2, 3, height, width, 3)
    else:
        values = np.frombuffer(raw, dtype='<f4', count=count).reshape(2, 3, height, width, 3)
    masks = np.unpackbits(np.frombuffer(raw, dtype=np.uint8, offset=floats_bytes),
        count=masks_count, bitorder='little').reshape(2, 3, height, width).astype(bool)
    return values, masks


def write_records(path, values, masks, metadata, *, shuffle=True):
    """Write phase-major arrays; caller owns atomic publication and locking."""
    path = Path(path)
    n, views, fields, h, w, xyz = values.shape
    if (views, fields, xyz) != (2, 3, 3) or masks.shape != (n, 2, 3, h, w):
        raise ValueError('Invalid motion record shapes')
    meta = dict(metadata, schema=SCHEMA, frames=n, height=h, width=w,
        cameras=list(CAMERAS), fields=['point_map','flow_dt1','flow_dt4'],
        dtype='<f4', source_stride=4, order='phase_major', shuffle=bool(shuffle),
        mask_packing='little_bitorder', compression='lz4_frame_level0_checksum')
    header = json.dumps(meta, sort_keys=True, separators=(',', ':')).encode()
    offsets = np.empty(n + 1, dtype='<u8')
    with path.open('w+b') as f:
        f.write(MAGIC + struct.pack('<Q', len(header)) + header)
        index_pos = f.tell()
        f.write(bytes((n + 1) * 8))
        for i in range(n):
            offsets[i] = f.tell()
            f.write(encode_record(values[i], masks[i], shuffle))
        offsets[n] = f.tell()
        f.seek(index_pos)
        f.write(offsets.tobytes())
        f.flush()
        os.fsync(f.fileno())
    return meta


class MotionRecords:
    def __init__(self, path):
        self.path = str(path)
        self.fd = os.open(path, os.O_RDONLY)
        try:
            prefix = os.pread(self.fd, 16, 0)
            if len(prefix) != 16 or prefix[:8] != MAGIC:
                raise ValueError('Invalid motion record magic/header')
            size = struct.unpack('<Q', prefix[8:])[0]
            if not 0 < size < 1024 * 1024:
                raise ValueError('Invalid motion header length')
            self.meta = json.loads(os.pread(self.fd, size, 16))
            if self.meta['schema'] != SCHEMA or self.meta.get('source_stride') != 4:
                raise ValueError('Unsupported motion record schema')
            if (self.meta.get('dtype') != '<f4' or self.meta.get('order') != 'phase_major'
                    or self.meta.get('cameras') != list(CAMERAS)
                    or self.meta.get('fields') != ['point_map', 'flow_dt1', 'flow_dt4']
                    or self.meta.get('mask_packing') != 'little_bitorder'):
                raise ValueError('Incompatible motion record layout')
            self.n = int(self.meta['frames'])
            self.h, self.w = int(self.meta['height']), int(self.meta['width'])
            if not 0 < self.n < 1000000 or (self.h, self.w) != (224, 224):
                raise ValueError('Unsupported motion dimensions')
            raw = os.pread(self.fd, (self.n + 1) * 8, 16 + size)
            self.offsets = np.frombuffer(raw, dtype='<u8')
            expected_start = 16 + size + (self.n + 1) * 8
            if (len(self.offsets) != self.n + 1 or self.offsets[0] != expected_start
                    or self.offsets[-1] != os.fstat(self.fd).st_size
                    or not np.all(self.offsets[1:] > self.offsets[:-1])):
                raise ValueError('Invalid/truncated motion record index')
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def __del__(self):
        if getattr(self, 'fd', None) is not None:
            self.close()

    def read_rows(self, first, count):
        if not 0 <= first < first + count <= self.n:
            raise IndexError((first, count))
        begin, end = int(self.offsets[first]), int(self.offsets[first + count])
        blob = os.pread(self.fd, end - begin, begin)
        if len(blob) != end - begin:
            raise OSError('Truncated motion record read')
        values, masks = [], []
        for i in range(first, first + count):
            lo, hi = int(self.offsets[i]) - begin, int(self.offsets[i+1]) - begin
            v, m = decode_record(memoryview(blob)[lo:hi], self.h, self.w, self.meta['shuffle'])
            values.append(v); masks.append(m)
        return np.stack(values), np.stack(masks)

    def clip(self, start, length=8, stride=4, flow_dt=4, prediction_horizon=32,
             normalization_cameras=CAMERAS):
        if length != 8 or stride != 4 or flow_dt not in (1, 4):
            raise ValueError('Record fast path requires8 sources at stride4, dt1/4')
        if start < 0 or start + 28 >= self.n or prediction_horizon < 28:
            raise ValueError('Clip source window out of bounds')
        values, masks = self.read_rows(phase_position(start, self.n), 8)
        return normalize_clip(values, masks, start, self.n, flow_dt, prediction_horizon,
                              normalization_cameras=normalization_cameras)


def normalize_camera_maps(cameras, normalization_cameras):
    """Use only the supervised views' valid points to define the clip frame.

    Points are centered and divided by one mean-radius scale across all sampled
    times. Displacements use that same scale without subtracting the center.
    Both input fields are metric values; no VAE input/latent scaling occurs here.
    """
    selected = motion_cameras(normalization_cameras)
    center, scale = canonical_normalization(
        [cameras[key]['point_map'] for key in selected],
        [cameras[key]['valid_mask'] for key in selected])
    for v in cameras.values():
        v['point_map'] = ((v['point_map'] - center) / scale).astype(np.float32)
        v['scene_flow'] = (v['scene_flow'] / scale).astype(np.float32)
        v['point_map'][~v['valid_mask']] = 0
        v['scene_flow'][~v['flow_valid_mask']] = 0
    return center, scale


def normalize_clip(values, masks, start, n, flow_dt, horizon=32,
                   normalization_cameras=CAMERAS):
    requested = start + np.arange(8) * 4
    targets = requested + flow_dt
    valid_flow_time = (targets < n) & (targets <= start + horizon)
    cameras = {}
    field = 1 if flow_dt == 1 else 2
    for view, camera in enumerate(CAMERAS):
        cameras[camera] = dict(
            point_map=np.ascontiguousarray(values[:, view, 0]),
            scene_flow=np.ascontiguousarray(values[:, view, field]),
            valid_mask=np.array(masks[:, view, 0], dtype=bool, order='C', copy=True),
            flow_valid_mask=np.array(masks[:, view, field], dtype=bool, order='C', copy=True))
        cameras[camera]['flow_valid_mask'] &= valid_flow_time[:, None, None]
    center, scale = normalize_camera_maps(cameras, normalization_cameras)
    return dict(cameras=cameras, frame_index=requested, requested_frame_index=requested,
        flow_target_frame_index=targets, sample_stride=4, flow_dt=flow_dt,
        prediction_horizon=horizon, center=center, scale=scale, normalized=True,
        normalization_cameras=list(normalization_cameras))
