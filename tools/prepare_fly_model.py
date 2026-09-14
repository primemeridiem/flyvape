"""
Reduce the Sketchfab fly download to a web-sized GLB, numpy only.

    python tools/prepare_fly_model.py build/models/src_fly/scene.gltf build/models/fly_model.glb

Model: "Wildtype Female Drosophila Melanogaster" by mlykouretzos, CC-BY 4.0,
https://sketchfab.com/3d-models/wildtype-female-drosophila-melanogaster-3ba1adba62f34fe995413ee5e9cf3c25

What it does, in order:
  1. reads every triangle primitive, applies node transforms, merges them
  2. vertex clustering: snaps vertices to a grid, averages position and colour
     per cell, drops collapsed and duplicate triangles; the grid is refined
     until the triangle count is just under the target
  3. orients the fly for the page: body length along X with the head at -X
     (the head is where the red compound-eye vertices are), the eyes spread
     along Z, the back toward +Y (the legs hang toward -Y)
  4. recomputes smooth normals and writes one GLB with POSITION, NORMAL,
     COLOR_0 (uint8) and uint32 indices
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def accessor(g, blob, i):
    a = g["accessors"][i]
    v = g["bufferViews"][a["bufferView"]]
    dt = np.dtype(COMPONENT[a["componentType"]])
    n = NCOMP[a["type"]]
    off = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = v.get("byteStride", dt.itemsize * n)
    if stride == dt.itemsize * n:
        arr = np.frombuffer(blob, dtype=dt, count=a["count"] * n, offset=off).reshape(a["count"], n)
    else:
        raw = np.frombuffer(blob, dtype=np.uint8, count=a["count"] * stride, offset=off).reshape(a["count"], stride)
        arr = raw[:, : dt.itemsize * n].copy().view(dt).reshape(a["count"], n)
    arr = arr.astype(np.float64 if dt == np.float32 else arr.dtype)
    if a.get("normalized") and dt != np.float32:
        arr = arr / np.iinfo(dt).max
    return arr


def node_matrix(n):
    if "matrix" in n:
        return np.array(n["matrix"], dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    if "scale" in n:
        m = np.diag(list(n["scale"]) + [1.0]) @ m
    if "rotation" in n:
        x, y, z, w = n["rotation"]
        r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        rm = np.eye(4); rm[:3, :3] = r
        m = rm @ m
    if "translation" in n:
        t = np.eye(4); t[:3, 3] = n["translation"]
        m = t @ m
    return m


def load(path):
    path = Path(path)
    g = json.loads(path.read_text())
    blob = (path.parent / g["buffers"][0]["uri"]).read_bytes()
    P, C, F = [], [], []
    base = 0

    def walk(ni, parent):
        nonlocal base
        n = g["nodes"][ni]
        m = parent @ node_matrix(n)
        if "mesh" in n:
            for p in g["meshes"][n["mesh"]]["primitives"]:
                if p.get("mode", 4) != 4:
                    continue
                pos = accessor(g, blob, p["attributes"]["POSITION"])
                pos = (np.c_[pos, np.ones(len(pos))] @ m.T)[:, :3]
                col = accessor(g, blob, p["attributes"]["COLOR_0"])[:, :3] if "COLOR_0" in p["attributes"] \
                    else np.full((len(pos), 3), 0.6)
                idx = accessor(g, blob, p["indices"]).reshape(-1, 3).astype(np.int64) if "indices" in p \
                    else np.arange(len(pos)).reshape(-1, 3)
                P.append(pos); C.append(col); F.append(idx + base)
                base += len(pos)
        for c in n.get("children", []):
            walk(c, m)

    for root in g["scenes"][g.get("scene", 0)]["nodes"]:
        walk(root, np.eye(4))
    return np.concatenate(P), np.concatenate(C), np.concatenate(F)


def cluster(P, C, F, cells):
    lo, hi = P.min(0), P.max(0)
    size = (hi - lo).max() / cells
    key = np.floor((P - lo) / size).astype(np.int64)
    dims = key.max(0) + 1
    lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    uniq, inv = np.unique(lin, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    P2 = np.stack([np.bincount(inv, P[:, k]) for k in range(3)], 1) / cnt[:, None]
    C2 = np.stack([np.bincount(inv, C[:, k]) for k in range(3)], 1) / cnt[:, None]
    F2 = inv[F]
    ok = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    F2 = F2[ok]
    srt = np.sort(F2, 1)
    _, first = np.unique(srt, axis=0, return_index=True)
    F2 = F2[np.sort(first)]
    used = np.unique(F2)
    remap = np.full(len(P2), -1, dtype=np.int64); remap[used] = np.arange(len(used))
    return P2[used], C2[used], remap[F2]


def eye_mask(C):
    """The red compound eyes. MEASURED on this model: the brown body reaches a redness
    (red minus the larger of green and blue) of 0.26; the eyes are the only vertices above 0.32."""
    return (C[:, 0] - np.maximum(C[:, 1], C[:, 2])) > 0.32


def orient(P, C):
    centre = P.mean(0)
    X = P - centre
    w, v = np.linalg.eigh(np.cov(X.T))
    axes = v[:, np.argsort(w)[::-1]]                  # columns: longest first
    red = eye_mask(C)
    along = axes[:, 0]
    if red.sum() > 50 and (X[red] @ along).mean() > 0:
        along = -along                                # head (eyes) toward -X
    rest = axes[:, 1:]
    if red.sum() > 50:                                # eyes spread sideways: the lateral axis
        spread = [np.var(X[red] @ rest[:, k]) for k in range(2)]
        lateral = rest[:, int(np.argmax(spread))]
    else:
        lateral = rest[:, 0]
    up = np.cross(lateral, along)
    up /= np.linalg.norm(up)
    lateral = np.cross(along, up)
    R = np.stack([along, up, lateral])                # rows: new X, Y, Z
    Q = X @ R.T
    # legs hang below the body: the far tail of the vertical distribution is ventral
    y = Q[:, 1]
    if abs(np.percentile(y, 1)) < abs(np.percentile(y, 99)):
        Q[:, 1] *= -1; Q[:, 2] *= -1                  # a rotation, not a mirror
    return Q, int(red.sum())


def landmarks(Q, C):
    """Points the page fits the rig to, in the model's own coordinates."""
    red = eye_mask(C)
    eye = Q[red].mean(0)
    left, right = Q[red & (Q[:, 2] < eye[2])], Q[red & (Q[:, 2] >= eye[2])]
    span = float(abs(right[:, 2].mean() - left[:, 2].mean()))
    x2, x98 = np.percentile(Q[:, 0], [2, 98])
    length = float(x98 - x2)
    zext = float(np.ptp(Q[:, 2]))
    mid = np.abs(Q[:, 2] - eye[2]) < 0.12 * zext
    # thorax centre is about a quarter body length behind the eyes; its top is where a tether is glued
    tx = eye[0] + 0.25 * length
    band = mid & (np.abs(Q[:, 0] - tx) < 0.04 * length)
    thorax_top = [float(tx), float(Q[band, 1].max()), float(eye[2])]
    # the face: the most forward point near the midline at eye height
    face_band = mid & (np.abs(Q[:, 1] - eye[1]) < 0.15 * float(np.ptp(Q[:, 1])))
    fi = np.argmin(np.where(face_band, Q[:, 0], np.inf))
    face = Q[fi].tolist()
    # feet: the lowest vertices; their mean x, z is where the ball should touch
    low = Q[:, 1] <= np.percentile(Q[:, 1], 1.5)
    feet = [float(Q[low, 0].mean()), float(np.percentile(Q[:, 1], 0.5)), float(Q[low, 2].mean())]
    return {"eye_mid": eye.tolist(), "eye_span": span, "length": length, "thorax_top": thorax_top,
            "face": face, "feet": feet}


def normals(P, F):
    N = np.zeros_like(P)
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    return N


def write_glb(out, P, N, C, F, extras, fit=None):
    pos = P.astype("<f4"); nrm = N.astype("<f4")
    col = np.clip(np.c_[C, np.ones(len(C))] * 255 + 0.5, 0, 255).astype(np.uint8)
    idx = F.astype("<u4")
    chunks, views, off = [], [], 0
    for arr, target in ((pos, 34962), (nrm, 34962), (col, 34962), (idx, 34963)):
        b = arr.tobytes(); pad = (-len(b)) % 4
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(b), "target": target})
        chunks.append(b + b"\0" * pad); off += len(b) + pad
    bin_ = b"".join(chunks)
    g = {
        "asset": {"version": "2.0", "generator": "flyvape prepare_fly_model.py", "extras": extras},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "fly", "extras": {"fit": fit or {}}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2}, "indices": 3, "material": 0}]}],
        "materials": [{"doubleSided": True, "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.6}}],
        "buffers": [{"byteLength": len(bin_)}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(pos), "type": "VEC3",
             "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": len(nrm), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5121, "normalized": True, "count": len(col), "type": "VEC4"},
            {"bufferView": 3, "componentType": 5125, "count": idx.size, "type": "SCALAR"},
        ],
    }
    js = json.dumps(g, separators=(",", ":")).encode(); js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(bin_)
    Path(out).write_bytes(struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
                          + struct.pack("<I4s", len(bin_), b"BIN\0") + bin_)
    return total


def main(src, out, target=80000):
    P, C, F = load(src)
    print(f"source: {len(P):,} vertices, {len(F):,} triangles")
    lo_c, hi_c, best = 40, 1200, None
    while hi_c - lo_c > 4:                            # the finest grid that stays under target
        mid = (lo_c + hi_c) // 2
        r = cluster(P, C, F, mid)
        if len(r[2]) <= target:
            best, lo_c = (mid, r), mid
        else:
            hi_c = mid
    cells, (P2, C2, F2) = best if best else (lo_c, cluster(P, C, F, lo_c))
    Q, nred = orient(P2, C2)
    N = normals(Q, F2)
    fit = landmarks(Q, C2)
    print("landmarks:", {k: [round(x, 3) for x in v] if isinstance(v, list) else round(v, 3) for k, v in fit.items()})
    extras = {"source": "Wildtype Female Drosophila Melanogaster by mlykouretzos",
              "license": "CC-BY-4.0", "url": "https://sketchfab.com/3d-models/wildtype-female-drosophila-melanogaster-3ba1adba62f34fe995413ee5e9cf3c25",
              "modified": "merged, vertex-clustered, re-oriented, normals recomputed"}
    size = write_glb(out, Q, N, C2, F2, extras, fit)
    ext = Q.max(0) - Q.min(0)
    print(f"grid {cells}: {len(Q):,} vertices, {len(F2):,} triangles, {size / 1e6:.2f} MB -> {out}")
    print(f"extent x {ext[0]:.3f} y {ext[1]:.3f} z {ext[2]:.3f}; red eye vertices {nred:,}; mean colour {C2.mean(0).round(3).tolist()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 80000)
