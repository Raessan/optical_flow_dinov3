import numpy as np
from PIL import Image
from os.path import *
import re

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TAG_CHAR = np.array([202021.25], np.float32)

def fill_nans_nearest(flow):
    """In-place nearest-neighbor fill for NaNs in a small set of pixels."""
    f = flow.copy()
    H, W, C = f.shape
    nan_any = np.isnan(f).any(axis=2)
    if not nan_any.any():
        return f

    # Precompute valid coords and their values
    ys_valid, xs_valid = np.where(~nan_any)
    valid_coords = np.stack([ys_valid, xs_valid], axis=1)

    # KD-like nearest via brute force (fine for a handful of NaNs)
    ys_nan, xs_nan = np.where(nan_any)
    for y, x in zip(ys_nan, xs_nan):
        dy = ys_valid - y
        dx = xs_valid - x
        idx = np.argmin(dy*dy + dx*dx)
        yy, xx = valid_coords[idx]
        f[y, x, :] = f[yy, xx, :]
    return f

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return fill_nans_nearest(np.resize(data, (int(h), int(w), 2)))
        
# def readFlow(fn, *, strict=True, nan_for_unknown=False):
#     """
#     Read .flo file in Middlebury format.

#     Params
#     ------
#     fn : str
#         Path to .flo file
#     strict : bool
#         If True, raise errors on malformed files (recommended).
#         If False, attempt best-effort reading.
#     nan_for_unknown : bool
#         If True, convert very large magnitudes (>= 1e9) to NaN (some writers
#         use large sentinels for unknown flow).
#     """
#     with open(fn, 'rb') as f:
#         # Read and validate magic tag (float32)
#         magic_arr = np.fromfile(f, np.float32, count=1)
#         if magic_arr.size != 1:
#             if strict:
#                 raise ValueError(f"{fn}: cannot read magic tag")
#             return None
#         magic = magic_arr.item()

#         if magic != TAG_CHAR:
#             # Try byteswapped tag to detect endianness mismatch
#             if np.float32(magic).byteswap().newbyteorder() == TAG_CHAR:
#                 # Endianness mismatch: we can proceed by marking that we need to byteswap the rest
#                 need_byteswap = True
#             else:
#                 if strict:
#                     raise ValueError(f"{fn}: bad magic {magic} (expected {TAG_CHAR})")
#                 print(f"Warning: {fn} has incorrect magic ({magic}).")
#                 return None
#         else:
#             need_byteswap = False

#         # Read width & height (int32)
#         w_arr = np.fromfile(f, np.int32, count=1)
#         h_arr = np.fromfile(f, np.int32, count=1)
#         if w_arr.size != 1 or h_arr.size != 1:
#             if strict:
#                 raise ValueError(f"{fn}: cannot read width/height")
#             return None

#         w, h = int(w_arr.item()), int(h_arr.item())

#         if w <= 0 or h <= 0:
#             if strict:
#                 raise ValueError(f"{fn}: invalid dimensions w={w}, h={h}")
#             print(f"Warning: {fn} invalid dimensions w={w}, h={h}")
#             return None

#         # Read payload
#         expected = 2 * w * h
#         data = np.fromfile(f, np.float32, count=expected)
#         if data.size != expected:
#             if strict:
#                 raise ValueError(
#                     f"{fn}: truncated payload: got {data.size} floats, expected {expected}"
#                 )
#             # Best-effort: pad with NaNs to expected length
#             pad = np.full(expected - data.size, np.nan, dtype=np.float32)
#             data = np.concatenate([data, pad])

#         if need_byteswap:
#             data = data.byteswap().newbyteorder()

#         # Reshape (NOT resize)
#         flow = data.reshape(h, w, 2)

#         # Optional: convert sentinel "unknown" to NaN
#         if nan_for_unknown:
#             # Many datasets mark unknown with huge numbers, e.g., >= 1e9
#             mask = np.abs(flow) >= 1e9
#             if mask.any():
#                 flow[mask] = np.nan

#         # Quick debug report if NaNs are present
#         if np.isnan(flow).any():
#             n_nans = int(np.isnan(flow).sum())
#             print(f"Debug: {fn} contains {n_nans} NaNs "
#                   f"({n_nans / flow.size:.4%} of entries).")

#         return flow

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid

def readDispKITTI(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    valid = disp > 0.0
    flow = np.stack([-disp, np.zeros_like(disp)], -1)
    return flow, valid


def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])
    

def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    return []