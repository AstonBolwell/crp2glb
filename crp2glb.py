"""
crp2glb.py - Unified CRP ripper

Usage:
    python crp2glb.py

Reads every .crp in ./input/, writes to ./output/<crp_stem>/:
    - *.glb               (meshes with embedded materials + textures)
    - textures/*.dds      (all extracted DDS textures)
    - meta.json           (summary of everything found/linked)
"""

import os
import io
import re
import json
import struct
import shutil
import tempfile
import traceback

# pip install pygltflib pillow
import pygltflib
from pygltflib import (
    GLTF2, Scene, Node, Mesh as GLTFMesh, Primitive, Attributes,
    Material, PbrMetallicRoughness, TextureInfo, Texture, Image, Sampler,
    Accessor, BufferView, Buffer,
)
from PIL import Image as PILImage

# ----------------------------- config -----------------------------
INPUT_DIR   = r".\input"
OUTPUT_ROOT = r".\output"

MIRROR_AXIS  = "x"
FLIP_WINDING = True
TYPE_SUFFIX  = b"PublicKeyToken=null"

# Cities: Skylines uses (0,0,0) to mean "cut out" and (8,8,8) to mean
# "intentionally black but visible". Treat anything with all channels
# below this threshold as fully transparent.
BLACK_CUTOFF = 1

# ----------------------------- helpers -----------------------------
def u8(f):  return struct.unpack("<B", f.read(1))[0]
def u16(f): return struct.unpack("<H", f.read(2))[0]
def u32(f): return struct.unpack("<I", f.read(4))[0]
def u64(f): return struct.unpack("<Q", f.read(8))[0]

def read_lp_string(f):
    n = u8(f)
    return f.read(n).decode("utf-8", "replace")

def safe_name(s):
    out = []
    for ch in s:
        out.append(ch if (ch.isalnum() or ch in "._- ()[]") else "_")
    return "".join(out).strip() or ""

def read_7bit_len_int(data, p):
    result = 0; shift = 0
    while True:
        b = data[p]; p += 1
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
    return result, p

def read_string(data, p):
    n, p = read_7bit_len_int(data, p)
    return data[p:p+n].decode("utf-8", "replace"), p + n

def sniff_type_string(payload):
    head = payload[:64]
    run_start = None
    for i, b in enumerate(head):
        printable = 32 <= b < 127
        if printable:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= 20:
                return head[run_start:i].decode("ascii", "replace")
            run_start = None
    if run_start is not None and len(head) - run_start >= 20:
        return head[run_start:].decode("ascii", "replace")
    return None

# ----------------------------- step 1: CRX extractor -----------------------------
TYPE_EXT = {
    "UnityEngine.Mesh":       ".mesh.bin",
    "UnityEngine.Material":   ".material.bin",
    "UnityEngine.GameObject": ".gameobject.bin",
    "VVehicleInfoGen":        ".vehicleinfo.bin",
    "VVehicleInfoSub":        ".vehicleinfosub.bin",
}

def classify(payload):
    png = payload.find(b"\x89PNG\r\n\x1a\n")
    if 0 <= png < 256:
        return "png", ".png", png
    dds = payload.find(b"DDS ")
    if 0 <= dds < 256:
        return "dds", ".dds", dds
    ts = sniff_type_string(payload)
    if ts:
        cls = ts.split(",")[0].strip()
        ext = TYPE_EXT.get(cls, ".bin")
        kind = cls.split(".")[-1].lower()
        return kind, ext, None
    return "unknown", ".bin", None

def extract_crp(crp_path, work_dir):
    """Unpack a .crp into work_dir. Returns manifest."""
    with open(crp_path, "rb") as f:
        data = f.read()

    f = open(crp_path, "rb")
    magic = f.read(4).decode("ascii", "replace")
    assert magic.upper() == "CRAP", f"not a CRP: {magic!r}"
    version           = u16(f)
    steam_id          = read_lp_string(f)
    unknown1          = read_lp_string(f)
    unknown2          = u32(f)
    name_of_mod       = read_lp_string(f)
    number_of_files   = u32(f)
    end_header_offset = u64(f)

    headers = []
    for i in range(number_of_files):
        headers.append({
            "file_name": read_lp_string(f),
            "unknown4":  read_lp_string(f),
            "unknown5":  u32(f),
            "offset_from_header": u64(f),
            "file_size": u64(f),
        })
    pos_after_headers = f.tell()
    f.close()

    print(f"    header: v{version} mod={name_of_mod!r} files={number_of_files} "
          f"header_end={end_header_offset} size={len(data)}")
    if pos_after_headers != end_header_offset:
        print(f"    WARN: header ends at {pos_after_headers}, "
              f"end_header_offset={end_header_offset} "
              f"(delta {end_header_offset - pos_after_headers})")

    assets = []
    used = set()
    for i, h in enumerate(headers):
        abs_off = end_header_offset + h["offset_from_header"]
        size = h["file_size"]
        if abs_off + size > len(data):
            print(f"    [{i:02d}] SKIP out-of-range off={abs_off} size={size}")
            continue

        payload = data[abs_off:abs_off + size]
        kind, ext, magic_off = classify(payload)
        write_bytes = payload[magic_off:] if magic_off is not None else payload

        stem = safe_name(h["file_name"])
        if not stem:
            ts = sniff_type_string(payload)
            if ts:
                stem = f"{ts.split(',')[0].split('.')[-1].lower()}_{h['offset_from_header']:08x}"
            else:
                stem = f"asset_{i:03d}"

        out_name = stem + ext
        n = 1
        while out_name.lower() in used:
            n += 1
            out_name = f"{stem}_{n}{ext}"
        used.add(out_name.lower())

        with open(os.path.join(work_dir, out_name), "wb") as g:
            g.write(write_bytes)

        assets.append({
            "index": i,
            "file_name": h["file_name"],
            "offset_from_header": h["offset_from_header"],
            "absolute_offset": abs_off,
            "file_size": size,
            "written_name": out_name,
            "written_size": len(write_bytes),
            "kind": kind,
        })

    manifest = {
        "crp_version": version,
        "steam_id": steam_id,
        "name_of_mod": name_of_mod,
        "number_of_files": number_of_files,
        "end_header_offset": end_header_offset,
        "assets": assets,
    }
    return manifest

# ----------------------------- step 2: mesh parser -----------------------------
def parse_mesh_full(data):
    assert data[0] == 0x00 and data[1] == 0x54, "not a mesh payload"
    idx = data.find(TYPE_SUFFIX, 2)
    if idx < 0:
        raise ValueError("type string not found")
    p = idx + len(TYPE_SUFFIX)
    name_len = data[p]; p += 1
    name = data[p:p+name_len].decode("ascii", "replace"); p += name_len
    V = struct.unpack("<I", data[p:p+4])[0]; p += 4

    positions = [struct.unpack("<3f", data[p+i*12:p+i*12+12]) for i in range(V)]
    p += V * 12

    def try_array(p, elem_size):
        if p + 4 > len(data): return 0, p
        count = struct.unpack("<I", data[p:p+4])[0]
        if count > V * 2 or count < 0: return 0, p
        if p + 4 + count * elem_size > len(data): return 0, p
        return count, p + 4 + count * elem_size

    color_count, p = try_array(p, 16)
    uv_count, p = try_array(p, 8);    uv_start = p - uv_count * 8
    norm_count, p = try_array(p, 12); norm_start = p - norm_count * 12
    tan_count, p = try_array(p, 16)
    bw_count, p = try_array(p, 32)
    bp_count, p = try_array(p, 64)

    uvs = [struct.unpack("<2f", data[uv_start+i*8:uv_start+i*8+8]) for i in range(uv_count)] if uv_count else []
    normals = [struct.unpack("<3f", data[norm_start+i*12:norm_start+i*12+12]) for i in range(norm_count)] if norm_count else []

    submesh_count = struct.unpack("<I", data[p:p+4])[0]; p += 4
    all_indices = []
    submesh_tri_counts = []
    for _ in range(submesh_count):
        tc = struct.unpack("<I", data[p:p+4])[0]; p += 4
        submesh_tri_counts.append(tc // 3)
        all_indices.extend(struct.unpack("<" + "I" * tc, data[p:p+tc*4])); p += tc * 4

    return {
        "name": name, "vertex_count": V,
        "positions": positions, "uvs": uvs, "normals": normals,
        "indices": all_indices,
        "submesh_count": submesh_count,
        "submesh_tri_counts": submesh_tri_counts,
    }

# ----------------------------- step 3: material parser -----------------------------
def parse_material(data):
    if data[0] != 0x00 or data[1] != 0x58:
        raise ValueError(f"not a material payload: {data[:4]!r}")
    idx = data.find(TYPE_SUFFIX, 2)
    if idx < 0: raise ValueError("type string not found")
    p = idx + len(TYPE_SUFFIX)
    name_len = data[p]; p += 1
    name = data[p:p+name_len].decode("ascii", "replace"); p += name_len

    shader, p = read_string(data, p)
    num_props = struct.unpack("<I", data[p:p+4])[0]; p += 4
    props = []
    for _ in range(num_props):
        ptype = struct.unpack("<I", data[p:p+4])[0]; p += 4
        pname, p = read_string(data, p)
        if ptype in (0, 1):
            v = struct.unpack("<4f", data[p:p+16]); p += 16
        elif ptype == 2:
            v = struct.unpack("<f", data[p:p+4])[0]; p += 4
        elif ptype == 3:
            isnull = data[p] != 0; p += 1
            v = "" if isnull else None
            if not isnull:
                v, p = read_string(data, p)
        else:
            v = None
        props.append((pname, ptype, v))
    return name, shader, props

# ----------------------------- texture helpers -----------------------------
def dds_to_png(dds_path):
    img = PILImage.open(dds_path).convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def maintex_with_black_cutout(maintex_path, cutoff=8):
    """Return PNG bytes where pure-black pixels in _MainTex become fully
    transparent. Cities: Skylines uses (0,0,0) for cut-outs and (8,8,8) for
    intentionally-black details, so the threshold must be strictly below 8."""
    img = PILImage.open(maintex_path).convert("RGBA")
    r, g, b, _ = img.split()

    # alpha = 0 if r<threshold AND g<threshold AND b<threshold, else 255
    # We use a minimum-of-channels image, then threshold it.
    from PIL import ImageChops
    mn = ImageChops.darker(ImageChops.darker(r, g), b)
    # mn has the minimum of the three channels per pixel.
    # Pixels with mn < cutoff are near-black -> transparent (0)
    # Pixels with mn >= cutoff are visible            -> opaque  (255)
    alpha = mn.point(lambda v: 0 if v < cutoff else 255)

    rgba = PILImage.merge("RGBA", (r, g, b, alpha))
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return buf.getvalue()

# ----------------------------- GLB builder -----------------------------
def build_glb(mesh, gltf_mat_source, work_dir, assets, out_path,
              mirror_axis="x", flip_winding=True):
    positions = mesh["positions"]
    mesh_uvs  = mesh["uvs"]
    mesh_nrm  = mesh["normals"]
    indices   = mesh["indices"]

    if mirror_axis in ("x", "y", "z"):
        ai = {"x": 0, "y": 1, "z": 2}[mirror_axis]
        positions = [(p[0] if ai != 0 else -p[0],
                      p[1] if ai != 1 else -p[1],
                      p[2] if ai != 2 else -p[2]) for p in positions]
        if mesh_nrm:
            mesh_nrm = [(n[0] if ai != 0 else -n[0],
                         n[1] if ai != 1 else -n[1],
                         n[2] if ai != 2 else -n[2]) for n in mesh_nrm]

    if flip_winding:
        new_idx = []
        for t in range(0, len(indices) - 2, 3):
            a, b, c = indices[t], indices[t+1], indices[t+2]
            new_idx.extend([a, c, b])
        indices = new_idx

    gltf = GLTF2()
    gltf.asset.generator = "crp2glb"
    buffer_data = b""
    accessors = []
    buffer_views = []

    def add_accessor(raw, comp_type, type_str, count, target=None):
        nonlocal buffer_data
        bv = BufferView(byteOffset=len(buffer_data), byteLength=len(raw), buffer=0)
        if target is not None:
            bv.target = target
        buffer_views.append(bv)
        buffer_data += raw
        while len(buffer_data) % 4:
            buffer_data += b"\x00"
        acc = Accessor(bufferView=len(buffer_views) - 1, componentType=comp_type,
                       count=count, type=type_str)
        accessors.append(acc)
        return len(accessors) - 1

    pos_flat = [c for v in positions for c in v]
    uv_flat  = [c for v in mesh_uvs for c in v]
    nrm_flat = [c for v in mesh_nrm for c in v]

    acc_pos = add_accessor(struct.pack("<" + "f"*len(pos_flat), *pos_flat), 5126, "VEC3", len(positions), 34962)
    acc_uv  = add_accessor(struct.pack("<" + "f"*len(uv_flat), *uv_flat), 5126, "VEC2", len(mesh_uvs), 34962) if mesh_uvs else None
    acc_nrm = add_accessor(struct.pack("<" + "f"*len(nrm_flat), *nrm_flat), 5126, "VEC3", len(mesh_nrm), 34962) if mesh_nrm else None
    acc_idx = add_accessor(struct.pack("<" + "I"*len(indices), *indices), 5125, "SCALAR", len(indices), 34963)

    images = []
    textures = []
    tex_indices = {}
    tex_records = []
    if gltf_mat_source:
        tex_paths = {}
        tex_assets = {}
        for pname, dds_asset in gltf_mat_source["tex_props"]:
            cs = dds_asset.strip().split()[-1] if dds_asset.strip() else ""
            a = find_asset_by_checksum(assets, cs)
            if not a: continue
            p = os.path.join(work_dir, a["written_name"])
            if not os.path.isfile(p): continue
            tex_paths[pname] = p
            tex_assets[pname] = a

        for pname, dds_path in tex_paths.items():
            try:
                if pname == "_MainTex":
                    png = maintex_with_black_cutout(dds_path, BLACK_CUTOFF)
                else:
                    png = dds_to_png(dds_path)
            except Exception as e:
                print(f"      image conversion failed for {pname}: {e}")
                try:
                    png = dds_to_png(dds_path)
                except Exception as e2:
                    print(f"      fallback also failed: {e2}")
                    continue

            img = Image(name=pname, mimeType="image/png", bufferView=len(buffer_views))
            bv = BufferView(byteOffset=len(buffer_data), byteLength=len(png), buffer=0)
            buffer_views.append(bv)
            buffer_data += png
            while len(buffer_data) % 4:
                buffer_data += b"\x00"
            images.append(img)
            tex_indices[pname] = len(images) - 1
            textures.append(Texture(source=len(images)-1, sampler=0))
            tex_records.append({
                "slot": pname,
                "dds_asset_index": tex_assets[pname]["index"],
                "dds_written_name": tex_assets[pname]["written_name"],
            })

    gltf_mat = None
    if "_MainTex" in tex_indices:
        gltf_mat = Material(
            name=f"mat_{gltf_mat_source['asset']['index']}" if gltf_mat_source else "mat",
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorTexture=TextureInfo(index=tex_indices["_MainTex"]),
                metallicFactor=0.0, roughnessFactor=1.0,
            ),
            doubleSided=True,
            alphaMode="MASK",
            alphaCutoff=0.5,
        )

    prim_attrs = Attributes(POSITION=acc_pos)
    if acc_uv is not None:  prim_attrs.TEXCOORD_0 = acc_uv
    if acc_nrm is not None: prim_attrs.NORMAL = acc_nrm

    prim = Primitive(attributes=prim_attrs, indices=acc_idx)
    if gltf_mat:
        prim.material = 0

    gltf.meshes = [GLTFMesh(name=mesh["name"] or "mesh", primitives=[prim])]
    if gltf_mat:
        gltf.materials = [gltf_mat]
    gltf.nodes = [Node(mesh=0)]
    gltf.scenes = [Scene(nodes=[0])]
    gltf.scene = 0
    gltf.accessors = accessors
    gltf.bufferViews = buffer_views
    gltf.images = images
    gltf.textures = textures
    gltf.samplers = [Sampler(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497)]
    gltf.buffers = [Buffer(byteLength=len(buffer_data))]
    gltf.set_binary_blob(buffer_data)
    gltf.save_binary(out_path)
    return tex_records

def find_asset_by_checksum(assets, checksum):
    needle = " - " + checksum
    for a in assets:
        if a.get("file_name", "").endswith(needle):
            return a
    return None

# ----------------------------- material matching -----------------------------
def material_score(m, assets):
    s = 0
    for pname, v in m["tex_props"]:
        cs = v.strip().split()[-1] if v.strip() else ""
        a = find_asset_by_checksum(assets, cs)
        if not a: continue
        if pname == "_MainTex": s += 100 + a["file_size"] // 1000
        elif pname == "_ACIMap": s += 10
        elif pname == "_XYSMap": s += 5
    if "Particle" in m["shader"] or "Additive" in m["shader"]: s -= 50
    if "Material" not in m["asset"]["file_name"]: s -= 10
    return s

def gather_materials(assets, work_dir):
    parsed = []
    for ma in assets:
        if ma.get("kind") != "material":
            continue
        path = os.path.join(work_dir, ma["written_name"])
        if not os.path.isfile(path):
            continue
        try:
            mname, shader, props = parse_material(open(path, "rb").read())
            tex_props = [(n, v) for n, t, v in props if t == 3 and v]
            parsed.append({
                "asset": ma, "name": mname, "shader": shader,
                "props": props, "tex_props": tex_props,
            })
        except Exception as e:
            print(f"    !! material [{ma['index']}] failed: {e}")
    return parsed

# ----------------------------- main pipeline -----------------------------
def process_crp(crp_path, out_dir):
    stem = os.path.splitext(os.path.basename(crp_path))[0]
    print(f"\n=== {stem} ===")
    print(f"  input:  {crp_path}")
    print(f"  output: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    tex_out = os.path.join(out_dir, "textures")
    os.makedirs(tex_out, exist_ok=True)

    work_dir = tempfile.mkdtemp(prefix="crp2glb_")
    try:
        print("  [1/3] extracting...")
        manifest = extract_crp(crp_path, work_dir)
        assets = manifest["assets"]
        assets_by_index = {a["index"]: a for a in assets}

        dds_assets = [a for a in assets if a["kind"] == "dds"]
        for a in dds_assets:
            src = os.path.join(work_dir, a["written_name"])
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(tex_out, a["written_name"]))
        print(f"  [2/3] copied {len(dds_assets)} DDS to textures/")

        print("  [3/3] building GLBs...")
        mats = gather_materials(assets, work_dir)
        train_mats = [m for m in mats if "Vehicle" in m["shader"]] or mats
        train_mats.sort(key=lambda m: -material_score(m, assets))
        best_mat = train_mats[0] if train_mats else None
        if best_mat:
            print(f"    material pool: {len(train_mats)} "
                  f"(best={best_mat['name']!r} shader={best_mat['shader']!r})")
        else:
            print("    material pool: none (GLBs will be untextured)")

        mesh_assets = [a for a in assets if a.get("kind") == "mesh"]
        glb_records = []
        for a in mesh_assets:
            src = os.path.join(work_dir, a["written_name"])
            if not os.path.isfile(src):
                continue
            try:
                mesh = parse_mesh_full(open(src, "rb").read())
            except Exception as e:
                print(f"    [{a['index']:02d}] !! mesh parse failed: {e}")
                continue

            base = safe_name(mesh["name"]) if mesh["name"] else f"mesh_{a['index']:02d}"
            base = f"{a['index']:02d}_{base}"
            out_path = os.path.join(out_dir, base + ".glb")

            try:
                tex_records = build_glb(mesh, best_mat, work_dir, assets,
                                        out_path, MIRROR_AXIS, FLIP_WINDING)
            except Exception as e:
                print(f"    [{a['index']:02d}] !! glb build failed: {e}")
                traceback.print_exc()
                continue

            xs = [p[0] for p in mesh["positions"]]
            ys = [p[1] for p in mesh["positions"]]
            zs = [p[2] for p in mesh["positions"]]
            bbox = [max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)] if xs else [0,0,0]

            print(f"    [{a['index']:02d}] {base}.glb  "
                  f"V={mesh['vertex_count']} tris={len(mesh['indices'])//3} "
                  f"textures={len(tex_records)}")

            glb_records.append({
                "mesh_asset_index": a["index"],
                "mesh_asset_written_name": a["written_name"],
                "mesh_internal_name": mesh["name"],
                "glb_file": os.path.basename(out_path),
                "vertex_count": mesh["vertex_count"],
                "triangle_count": len(mesh["indices"]) // 3,
                "submesh_count": mesh["submesh_count"],
                "submesh_triangle_counts": mesh["submesh_tri_counts"],
                "bbox": bbox,
                "material_used": best_mat["name"] if best_mat else None,
                "shader_used": best_mat["shader"] if best_mat else None,
                "textures_embedded": tex_records,
            })

        materials_summary = []
        for m in mats:
            materials_summary.append({
                "asset_index": m["asset"]["index"],
                "written_name": m["asset"]["written_name"],
                "name": m["name"],
                "shader": m["shader"],
                "score": material_score(m, assets),
                "texture_slots": [
                    {"slot": pname, "raw_ref": v} for pname, v in m["tex_props"]
                ],
                "property_count": len(m["props"]),
            })

        kind_counts = {}
        for a in assets:
            kind_counts[a["kind"]] = kind_counts.get(a["kind"], 0) + 1

        meta = {
            "crp_file": os.path.basename(crp_path),
            "crp_version": manifest["crp_version"],
            "steam_id": manifest["steam_id"],
            "name_of_mod": manifest["name_of_mod"],
            "asset_count": len(assets),
            "asset_kind_counts": kind_counts,
            "generated_glbs": glb_records,
            "materials": materials_summary,
            "dds_textures": [
                {
                    "asset_index": a["index"],
                    "file_name": a["file_name"],
                    "written_name": a["written_name"],
                    "size": a["file_size"],
                }
                for a in dds_assets
            ],
            "all_assets": [
                {
                    "index": a["index"],
                    "file_name": a["file_name"],
                    "kind": a["kind"],
                    "written_name": a["written_name"],
                    "file_size": a["file_size"],
                }
                for a in assets
            ],
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as g:
            json.dump(meta, g, indent=2, ensure_ascii=False)

        print(f"  done: {len(glb_records)} GLB(s), {len(dds_assets)} texture(s), "
              f"meta.json")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    crps = sorted(
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".crp")
    )
    if not crps:
        print(f"No .crp files in {INPUT_DIR}")
        return

    print(f"Found {len(crps)} .crp file(s)")
    for crp in crps:
        stem = os.path.splitext(os.path.basename(crp))[0]
        out_dir = os.path.join(OUTPUT_ROOT, stem)
        try:
            process_crp(crp, out_dir)
        except Exception as e:
            print(f"!! FAILED {stem}: {e}")
            traceback.print_exc()

    print("\nAll done.")


if __name__ == "__main__":
    main()
