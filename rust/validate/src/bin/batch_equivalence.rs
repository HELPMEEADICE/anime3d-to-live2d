// SPDX-License-Identifier: MIT
// Exact vertex/surface correspondence after adjacent ArtMesh concatenation.
use anyhow::{Context, Result, ensure};
use ayagami::{
    core::{ArtMesh, ArtMeshForm, BlendFormMap, Item, Model, Param},
    driver::Driver,
    file,
};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    env, fs,
    fs::File,
    path::Path,
};

fn load(path: &Path) -> Result<file::ParsedModel> {
    let m: Value = serde_json::from_slice(&fs::read(path)?)?;
    Ok(file::ParsedModel::load(&mut File::open(
        path.parent()
            .unwrap()
            .join(m["FileReferences"]["Moc"].as_str().context("Moc")?),
    )?)?)
}
fn page_opaque(path: &Path, page: u32) -> Result<bool> {
    let manifest: Value = serde_json::from_slice(&fs::read(path)?)?;
    let texture = path.parent().unwrap().join(
        manifest["FileReferences"]["Textures"][page as usize]
            .as_str()
            .context("Texture")?,
    );
    Ok(image::open(texture)?
        .to_rgba8()
        .pixels()
        .all(|p| p.0[3] == 255))
}
fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    ensure!(
        args.len() == 6,
        "batch_equivalence BEFORE AFTER MAPPING CASES REPORT"
    );
    let before = load(Path::new(&args[1]))?;
    let after = load(Path::new(&args[2]))?;
    let am: Vec<_> = before.artmeshes().into_iter().collect();
    let bm: Vec<_> = after.artmeshes().into_iter().collect();
    let aid: HashMap<_, _> = am.iter().map(|m| (m.id(), m)).collect();
    let bid: HashMap<_, _> = bm.iter().map(|m| (m.id(), m)).collect();
    let manifest: Value = serde_json::from_slice(&fs::read(&args[3])?)?;
    let mapping: Vec<Value> = if manifest.is_array() {
        serde_json::from_value(manifest.clone())?
    } else {
        serde_json::from_value(manifest["rows"].clone())?
    };
    let proxies: Vec<Value> = manifest["mask_proxies"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    ensure!(
        proxies.iter().all(|r| r["mask_proxy"] == true)
            && mapping.iter().all(|r| r["mask_proxy"] != true),
        "Invalid proxy row class"
    );
    let proxied_sources: HashSet<_> = proxies
        .iter()
        .map(|r| r["source"].as_str().unwrap())
        .collect();
    let extras: HashSet<String> = manifest["extra_unobservable"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    ensure!(mapping.len() == am.len(), "incomplete mapping");
    ensure!(
        mapping
            .iter()
            .map(|r| r["source"].as_str().unwrap())
            .collect::<HashSet<_>>()
            .len()
            == am.len(),
        "duplicate source mapping"
    );
    let zero_before: HashSet<_> = am
        .iter()
        .filter(|m| {
            m.forms().into_iter().all(|f| f.opacity() == 0.)
                && m.blend_form_maps()
                    .into_iter()
                    .flatten()
                    .all(|b| b.forms().into_iter().all(|f| f.opacity() == 0.))
        })
        .map(|m| m.id().to_string())
        .collect();
    let zero_after: HashSet<_> = bm
        .iter()
        .filter(|m| {
            m.forms().into_iter().all(|f| f.opacity() == 0.)
                && m.blend_form_maps()
                    .into_iter()
                    .flatten()
                    .all(|b| b.forms().into_iter().all(|f| f.opacity() == 0.))
        })
        .map(|m| m.id().to_string())
        .collect();
    let used: HashSet<_> = am
        .iter()
        .flat_map(|m| m.clips().into_iter().map(|s| s.id().to_string()))
        .collect();
    let mut expansion: HashMap<&str, Vec<(usize, &str)>> = HashMap::new();
    for row in mapping.iter().chain(&proxies) {
        if row["removed"] == true {
            continue;
        }
        expansion
            .entry(row["target"].as_str().unwrap())
            .or_default()
            .push((
                row["vertex_start"].as_u64().unwrap() as usize,
                row["source"].as_str().unwrap(),
            ));
    }
    for members in expansion.values_mut() {
        members.sort_by_key(|m| m.0);
    }
    for name in &extras {
        let mesh = bid[name.as_str()];
        ensure!(
            zero_after.contains(name)
                && mesh.clips().into_iter().next().is_none()
                && !bm
                    .iter()
                    .any(|m| m.clips().into_iter().any(|s| s.id() == name)),
            "observable extra carrier"
        );
    }
    ensure!(
        expansion.len() + extras.len() == bm.len()
            && bm
                .iter()
                .all(|m| expansion.contains_key(m.id()) || extras.contains(m.id())),
        "unmapped target mesh"
    );
    for (target, members) in &expansion {
        let mut cursor = 0usize;
        for &(start, source) in members {
            ensure!(start == cursor, "overlap or gap in target vertices");
            cursor += aid[source].vertex_count() as usize;
        }
        ensure!(
            cursor == bid[target].vertex_count() as usize,
            "extra target vertices"
        );
    }
    let contract = |m: &file::ParsedModel| {
        m.params()
            .into_iter()
            .map(|p| {
                (
                    p.id().to_string(),
                    p.min().to_bits(),
                    p.max().to_bits(),
                    p.default().to_bits(),
                    p.is_blendshape(),
                )
            })
            .collect::<Vec<_>>()
    };
    ensure!(contract(&before) == contract(&after), "parameters changed");
    let mut permutations: HashMap<&str, Vec<usize>> = HashMap::new();
    let resource_manifest: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let mut opaque_pages: HashMap<u32, bool> = HashMap::new();
    let opaque_union = manifest["opaque_mask_union"] == true;
    if opaque_union {
        let mut pages = HashSet::new();
        for m in &am {
            if used.contains(m.id()) && pages.insert(m.texture()) {
                ensure!(
                    page_opaque(Path::new(&args[1]), m.texture())?,
                    "Nonopaque union source"
                );
            }
        }
        let after_used: HashSet<_> = bm
            .iter()
            .flat_map(|m| m.clips().into_iter().map(|s| s.id().to_owned()))
            .collect();
        pages.clear();
        for m in &bm {
            if after_used.contains(m.id()) && pages.insert(m.texture()) {
                ensure!(
                    page_opaque(Path::new(&args[2]), m.texture())?,
                    "Nonopaque union proxy"
                );
            }
        }
    }
    for row in mapping.iter().chain(&proxies) {
        let name = row["source"].as_str().unwrap();
        let n = aid[name].vertex_count() as usize;
        let order: Vec<usize> = if let Some(v) = row["vertex_order"].as_array() {
            v.iter().map(|x| x.as_u64().unwrap() as usize).collect()
        } else {
            (0..n).collect()
        };
        ensure!(
            order.len() == n
                && order.iter().all(|&v| v < n)
                && order.iter().copied().collect::<HashSet<_>>().len() == n,
            "invalid vertex permutation"
        );
        let mut inverse = vec![0; n];
        for (new, &old) in order.iter().enumerate() {
            inverse[old] = new;
        }
        permutations.insert(name, inverse);
    }
    for row in mapping.iter().chain(&proxies) {
        let a = aid[row["source"].as_str().unwrap()];
        if row["removed"] == true {
            ensure!(
                zero_before.contains(a.id())
                    && (!used.contains(a.id()) || proxied_sources.contains(a.id()))
                    && a.clips().into_iter().next().is_none(),
                "removed observable mesh"
            );
            continue;
        }
        let b = bid[row["target"].as_str().unwrap()];
        let proxy = row["mask_proxy"] == true;
        if proxy {
            ensure!(
                zero_after.contains(b.id())
                    && b.clips().into_iter().next().is_none()
                    && used.contains(a.id()),
                "observable or unused mask proxy"
            );
        }
        if row["ignore_hidden_depth"] == true || row["ignore_hidden_color"] == true {
            ensure!(
                zero_before.contains(a.id())
                    && zero_after.contains(b.id())
                    && a.clips().into_iter().next().is_none(),
                "ignored observable depth"
            );
        }
        let vs = row["vertex_start"].as_u64().unwrap() as usize;
        let n = a.vertex_count() as usize;
        ensure!(
            (a.texture() == b.texture() || (proxy && opaque_union))
                && a.culling() == b.culling()
                && (proxy || a.invert_mask() == b.invert_mask()),
            "flags changed"
        );
        let mut old_clips = a
            .clips()
            .into_iter()
            .map(|m| m.id().to_owned())
            .collect::<Vec<_>>();
        let mut new_clips = b
            .clips()
            .into_iter()
            .flat_map(|m| expansion[m.id()].iter().map(|(_, name)| name.to_string()))
            .collect::<Vec<_>>();
        if opaque_union {
            old_clips.sort();
            new_clips.sort();
        }
        ensure!(proxy || old_clips == new_clips, "clipping changed");
        let au = a.texcoord_offset() as usize;
        let bu = b.texcoord_offset() as usize + vs;
        let permutation = &permutations[a.id()];
        for old in 0..n {
            let x = before.texcoord_buffer().unwrap()[au + old];
            let y = after.texcoord_buffer().unwrap()[bu + permutation[old]];
            ensure!(
                x.x.to_bits() == y.x.to_bits() && x.y.to_bits() == y.y.to_bits(),
                "UV changed"
            );
        }
        let ar = a.index_range();
        let br = b.index_range();
        let bi = br.start as usize + row["index_start"].as_u64().unwrap() as usize;
        let original = &before.index_buffer().unwrap()[ar.start as usize..ar.end as usize];
        let cache_reorder = row["opaque_mask_triangle_order"].is_array();
        let keep: Vec<usize> = if let Some(k) = row["opaque_mask_triangle_order"].as_array() {
            let page = a.texture();
            if !opaque_pages.contains_key(&page) {
                let path = Path::new(&args[1]).parent().unwrap().join(
                    resource_manifest["FileReferences"]["Textures"][page as usize]
                        .as_str()
                        .unwrap(),
                );
                opaque_pages.insert(
                    page,
                    image::open(path)?
                        .to_rgba8()
                        .pixels()
                        .all(|p| p.0[3] == 255),
                );
            }
            ensure!(
                zero_before.contains(a.id())
                    && zero_after.contains(b.id())
                    && used.contains(a.id())
                    && a.clips().into_iter().next().is_none()
                    && opaque_pages[&page],
                "reordered observable or translucent primitives"
            );
            let k: Vec<_> = k.iter().map(|v| v.as_u64().unwrap() as usize).collect();
            ensure!(
                k.len() == original.len() / 3
                    && k.iter().all(|&i| i < k.len())
                    && k.iter().copied().collect::<HashSet<_>>().len() == k.len(),
                "changed mask triangles"
            );
            k
        } else if let Some(k) = row["kept_triangles"].as_array() {
            k.iter().map(|v| v.as_u64().unwrap() as usize).collect()
        } else {
            (0..original.len() / 3).collect()
        };
        ensure!(
            cache_reorder || keep.windows(2).all(|w| w[0] < w[1]),
            "triangle order changed"
        );
        let kept_set: HashSet<_> = keep.iter().copied().collect();
        for (i, t) in original.chunks_exact(3).enumerate() {
            ensure!(
                kept_set.contains(&i) || t[0] == t[1] || t[1] == t[2] || t[0] == t[2],
                "removed nondegenerate triangle"
            );
        }
        let expected: Vec<_> = keep
            .iter()
            .flat_map(|&i| original[3 * i..3 * i + 3].iter().copied())
            .collect();
        ensure!(
            row["index_count"].as_u64().unwrap() as usize == expected.len(),
            "index extent mismatch"
        );
        for (&x, &y) in expected
            .iter()
            .zip(&after.index_buffer().unwrap()[bi..bi + expected.len()])
        {
            ensure!(
                permutation[x as usize] + vs == y as usize,
                "triangle changed"
            );
        }
    }
    let cases: Vec<Value> = serde_json::from_slice(&fs::read(&args[4])?)?;
    let mut a = Driver::new(&before);
    let mut b = Driver::new(&after);
    let mut count = 0u64;
    for (i, case) in cases.iter().enumerate() {
        for p in before.params() {
            a.set_param(p.uid(), p.default())?;
        }
        for p in after.params() {
            b.set_param(p.uid(), p.default())?;
        }
        for (p, v) in case["parameters"].as_object().context("parameters")? {
            a.set_param_by_id(p, v.as_f64().unwrap() as f32)?;
            b.set_param_by_id(p, v.as_f64().unwrap() as f32)?;
        }
        a.drive(&before);
        b.drive(&after);
        for row in mapping.iter().chain(&proxies) {
            let x = aid[row["source"].as_str().unwrap()];
            if row["removed"] == true {
                ensure!(
                    a.artmesh_state(x.uid()).unwrap().visual.opacity == 0.,
                    "removed mesh became visible"
                );
                continue;
            }
            let y = bid[row["target"].as_str().unwrap()];
            let sx = a.artmesh_state(x.uid()).unwrap();
            let sy = b.artmesh_state(y.uid()).unwrap();
            let start = row["vertex_start"].as_u64().unwrap() as usize;
            if row["mask_proxy"] == true {
                ensure!(
                    sx.visual.visible == sy.visual.visible && sy.visual.opacity == 0.,
                    "proxy visibility changed {} {}",
                    i,
                    x.id()
                );
            } else {
                ensure!(
                    (sx.depth == sy.depth || row["ignore_hidden_depth"] == true)
                        && sx.visual.visible == sy.visual.visible
                        && sx.visual.opacity.to_bits() == sy.visual.opacity.to_bits()
                        && ((sx.visual.multiply_color == sy.visual.multiply_color
                            && sx.visual.screen_color == sy.visual.screen_color)
                            || row["ignore_hidden_color"] == true),
                    "visual changed {} {}",
                    i,
                    x.id()
                );
            }
            for (old, vx) in sx.vertices.iter().enumerate() {
                let vy = &sy.vertices[start + permutations[x.id()][old]];
                ensure!(
                    vx.x.to_bits() == vy.x.to_bits() && vx.y.to_bits() == vy.y.to_bits(),
                    "vertex changed {} {}",
                    i,
                    x.id()
                );
                count += 1;
            }
        }
        if i % 100 == 0 {
            eprintln!("Compared {}/{}", i + 1, cases.len());
        }
    }
    fs::write(
        &args[5],
        serde_json::to_vec(
            &json!({"poses":cases.len(),"compared_vertices":count,"different_vertices":0,"before_meshes":am.len(),"after_meshes":bm.len(),"removed_unobservable_meshes":mapping.iter().filter(|r|r["removed"]==true).count(),"proxy_vertex_rows":proxies.len(),"retained_vertex_uv_visuals_equal":true,"mask_source_sequence_equal_after_expansion":!opaque_union,"mask_source_multiset_equal_after_expansion":true,"opaque_mask_union":opaque_union,"method":"Unmodified public driver; all retained and proxy vertex coordinates bit-exact. Main visual functions unchanged, apart from separately proven zero-opacity exceptions. UVs and every oriented primitive checked. Opaque-union mode directly checks all source/proxy texture alpha bytes are 255 and preserves expanded per-context source multisets; otherwise expanded source order is exact. A dropped original mask source must be permanently invisible, unclipped and retained in a verified proxy. GPU alpha checked separately."}),
        )?,
    )?;
    Ok(())
}
