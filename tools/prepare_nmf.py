"""
Build a web GLB of the NeuroMechFly v2 fruit fly with a node per body segment.

    python tools/prepare_nmf.py build/models/nmf build/models/fly_nmf.glb

Source: NeuroMechFly v2 via flygym (github.com/NeLy-EPFL/flygym), Apache-2.0,
Ramdya lab, EPFL. Files used: assets/model/neuromechfly/rigging.yaml,
pose/_manual_specs/neutral.yaml, visuals.yaml (colours copied below) and
meshes/simplified_max2000faces/*.stl. Changes made here: STL converted to
indexed glTF meshes in millimetres, right-side segments mirrored from the left
meshes (as flygym does), the neutral pose baked into node rotations, and the
MuJoCo frame (x forward, y left, z up) rotated to the page frame (head toward
-X, back toward +Y, left side toward +Z).

Each segment node carries extras:
    rest   [w, x, y, z]      rotation from rigging.yaml
    joints [[axis, radians]] neutral joint angles in application order (pitch=y, roll=z, yaw=x)
    side   "l" | "r" | "c"
so the page can re-pose any joint: rotation = rest * R(joint 1) * R(joint 2) * ..., in the listed order.
flygym's axes are pitch about y, roll about z and yaw about x; right-side roll and yaw
are mirrored (flygym flips their axis vectors, which is the same as negating the angle).
"""
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

COLOURS = {  # visuals.yaml, rgb1 of each set, alpha from rgba
    "wing": (0.8, 0.8, 0.9, 0.3), "eye": (0.67, 0.21, 0.12, 1.0), "arista": (0.26, 0.2, 0.16, 1.0),
    "haltere": (0.59, 0.43, 0.24, 1.0), "headthorax": (0.59, 0.39, 0.12, 1.0),
    "antennaproboscis": (0.59, 0.39, 0.12, 1.0), "abdomen": (0.70, 0.53, 0.30, 1.0),
    "abdomen6": (0.60, 0.43, 0.23, 1.0), "coxa": (0.59, 0.39, 0.12, 1.0),
    "trochanterfemur": (0.63, 0.43, 0.16, 1.0), "tibia": (0.67, 0.47, 0.2, 1.0), "tarsus": (0.71, 0.51, 0.24, 1.0),
}


def colour_set(body):
    b = body[2:] if body[1] == "_" else body
    if body.endswith("_wing"): return "wing"
    if body.endswith("_eye"): return "eye"
    if body.endswith("_arista"): return "arista"
    if body.endswith("_haltere"): return "haltere"
    if body in ("c_head", "c_thorax"): return "headthorax"
    if body.endswith(("_pedicel", "_funiculus")) or body in ("c_rostrum", "c_haustellum"): return "antennaproboscis"
    if body == "c_abdomen6": return "abdomen6"
    if body.startswith("c_abdomen"): return "abdomen"
    for k in ("coxa", "trochanterfemur", "tibia", "tarsus"):
        if k in body: return k
    return "headthorax"


def parse_rigging(path):
    bodies, cur = {}, None
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if not line.startswith(" ") and line.rstrip().endswith(":"):
            cur = line.strip()[:-1]
            bodies[cur] = {}
        elif cur:
            k, v = line.strip().split(":", 1)
            v = v.strip()
            bodies[cur][k] = [float(x) for x in v.strip("[]").split(",")] if v.startswith("[") else float(v)
    return bodies


def parse_pose(path):
    angles, order = {}, None
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s.startswith("axis_order:"):
            order = [a.strip() for a in s.split(":", 1)[1].strip().strip("[]").split(",")]
        m = re.match(r"^([\w]+)-([\w]+)-(roll|pitch|yaw):\s*(-?[\d.]+)", s)
        if m:
            angles[(m.group(2), m.group(3))] = np.deg2rad(float(m.group(4)))
    return angles, order or ["roll", "pitch", "yaw"]


def parent_of(b):
    if b == "c_thorax": return None
    if b == "c_head": return "c_thorax"
    if b == "c_rostrum": return "c_head"
    if b == "c_haustellum": return "c_rostrum"
    if b == "c_abdomen12": return "c_thorax"
    m = re.match(r"c_abdomen(\d)$", b)
    if m: return {"3": "c_abdomen12", "4": "c_abdomen3", "5": "c_abdomen4", "6": "c_abdomen5"}[m.group(1)]
    s = b[0]
    if b.endswith(("_eye", "_pedicel")): return "c_head"
    if b.endswith("_funiculus"): return f"{s}_pedicel"
    if b.endswith("_arista"): return f"{s}_funiculus"
    if b.endswith(("_haltere", "_wing")): return "c_thorax"
    leg, seg = b.split("_", 1)
    if seg == "coxa": return "c_thorax"
    if seg == "trochanterfemur": return f"{leg}_coxa"
    if seg == "tibia": return f"{leg}_trochanterfemur"
    if seg == "tarsus1": return f"{leg}_tibia"
    m = re.match(r"tarsus(\d)$", seg)
    return f"{leg}_tarsus{int(m.group(1)) - 1}"


def read_stl(p):
    b = Path(p).read_bytes()
    if b[:5].lower() == b"solid" and b"facet" in b[:400]:
        v = [list(map(float, l.split()[1:4])) for l in b.decode(errors="ignore").splitlines() if l.strip().startswith("vertex")]
        tri = np.array(v, dtype=np.float64).reshape(-1, 3, 3)
    else:
        n = struct.unpack("<I", b[80:84])[0]
        a = np.frombuffer(b, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
        tri = a["v"].astype(np.float64)
    v = tri.reshape(-1, 3) * 1000.0                                # metres -> mm
    key = np.round(v / 1e-4).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    P = v[first]; F = inv.reshape(-1, 3)
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    N = np.zeros_like(P)
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    return P.astype("<f4"), N.astype("<f4"), F


def main(src, out):
    src = Path(src)
    rig = parse_rigging(src / "rigging.yaml")
    pose, order = parse_pose(src / "pose" / "neutral_manual.yaml")
    axis = {"roll": "z", "pitch": "y", "yaw": "x"}   # flygym anatomy.RotationAxis: pitch->y, roll->z, yaw->x

    views, accessors, meshes, materials, chunks = [], [], [], [], []
    off = 0

    def add_view(arr, target):
        nonlocal off
        b = arr.tobytes(); pad = (-len(b)) % 4
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(b), "target": target})
        chunks.append(b + b"\0" * pad); off += len(b) + pad
        return len(views) - 1

    mat_index = {}
    for name, (r, g, b, a) in COLOURS.items():
        m = {"name": name, "pbrMetallicRoughness": {"baseColorFactor": [r, g, b, a], "metallicFactor": 0.0, "roughnessFactor": 0.6},
             "doubleSided": True}
        if a < 1: m["alphaMode"] = "BLEND"
        mat_index[name] = len(materials); materials.append(m)

    mesh_index, tri_total = {}, 0
    for stl in sorted((src / "meshes").glob("*.stl")):
        P, N, F = read_stl(stl)
        idx = F.astype("<u2" if len(P) < 65535 else "<u4")
        ap = len(accessors); accessors.append({"bufferView": add_view(P, 34962), "componentType": 5126, "count": len(P), "type": "VEC3",
                                               "min": P.min(0).tolist(), "max": P.max(0).tolist()})
        an = len(accessors); accessors.append({"bufferView": add_view(N, 34962), "componentType": 5126, "count": len(N), "type": "VEC3"})
        ai = len(accessors); accessors.append({"bufferView": add_view(idx.reshape(-1), 34963),
                                               "componentType": 5123 if idx.dtype == np.dtype("<u2") else 5125, "count": idx.size, "type": "SCALAR"})
        meshes.append({"name": stl.stem, "primitives": [{"attributes": {"POSITION": ap, "NORMAL": an}, "indices": ai,
                                                        "material": mat_index[colour_set(stl.stem)]}]})
        mesh_index[stl.stem] = len(meshes) - 1
        tri_total += len(F)

    def quat_mul(a, b):  # w, x, y, z
        w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
        return [w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2]

    def axis_quat(ax, ang):
        s, c = np.sin(ang / 2), np.cos(ang / 2)
        return [c, s if ax == "x" else 0.0, s if ax == "y" else 0.0, s if ax == "z" else 0.0]

    nodes = [{"name": "nmf_root", "rotation": [0.0, 0.70710678, 0.70710678, 0.0], "children": []}]
    node_of = {}
    names = list(rig.keys())
    for b in names:
        node_of[b] = len(nodes); nodes.append({"name": b})
    for b in names:
        spec = rig[b]
        side = b[0] if b[1] in "_fmh" and b[0] in "lr" else "c"
        left_name = ("l" + b[1:]) if side == "r" else b
        joints = []
        q = [float(x) for x in spec.get("quat", [1, 0, 0, 0])]
        rest = list(q)
        for ax_name in order:
            ang = pose.get((left_name, ax_name))
            if ang is None:
                continue
            if side == "r" and ax_name in ("roll", "yaw"):
                ang = -ang                                         # mirror across the midline
            joints.append([axis[ax_name], float(ang)])
            q = quat_mul(q, axis_quat(axis[ax_name], ang))
        n = nodes[node_of[b]]
        n["translation"] = [float(x) for x in spec["pos"]]
        n["rotation"] = [q[1], q[2], q[3], q[0]]                   # glTF x, y, z, w
        n["extras"] = {"rest": rest, "joints": joints, "side": side}
        mesh_name = left_name if left_name in mesh_index else None
        if mesh_name is not None:
            if side == "r":
                gi = len(nodes); nodes.append({"name": b + "_geom", "mesh": mesh_index[mesh_name], "scale": [1.0, -1.0, 1.0]})
                n.setdefault("children", []).append(gi)
            else:
                n["mesh"] = mesh_index[mesh_name]
        p = parent_of(b)
        if p is None:
            nodes[0]["children"].append(node_of[b])
        else:
            nodes[node_of[p]].setdefault("children", []).append(node_of[b])

    bin_ = b"".join(chunks)
    g = {"asset": {"version": "2.0", "generator": "flyvape tools/prepare_nmf.py",
                   "extras": {"source": "NeuroMechFly v2 (flygym, NeLy-EPFL), Apache-2.0", "units": "mm"}},
         "scene": 0, "scenes": [{"nodes": [0]}], "nodes": nodes, "meshes": meshes, "materials": materials,
         "buffers": [{"byteLength": len(bin_)}], "bufferViews": views, "accessors": accessors}
    js = json.dumps(g, separators=(",", ":")).encode(); js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(bin_)
    Path(out).write_bytes(struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
                          + struct.pack("<I4s", len(bin_), b"BIN\0") + bin_)
    print(f"{len(names)} segments, {len(meshes)} meshes, {tri_total:,} triangles, {total / 1e6:.2f} MB -> {out}")
    print("pose joints set:", sum(len(nodes[node_of[b]]['extras']['joints']) for b in names))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
