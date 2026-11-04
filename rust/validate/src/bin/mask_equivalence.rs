// SPDX-License-Identifier: MIT
// External verifier of mask-union factorization via the public unmodified driver.
use anyhow::{Context, Result, ensure};
use ayagami::{
    core::{ArtMesh, Item, Model, Param},
    driver::Driver,
    file,
};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    env,
    fs::{self, File},
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
fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    ensure!(
        args.len() == 6,
        "mask_equivalence BEFORE.model3.json AFTER.model3.json PLAN.json CASES.json REPORT.json"
    );
    let a = load(Path::new(&args[1]))?;
    let b = load(Path::new(&args[2]))?;
    let plan: Value = serde_json::from_slice(&fs::read(&args[3])?)?;
    let am: Vec<_> = a.artmeshes().into_iter().collect();
    let bm: Vec<_> = b.artmeshes().into_iter().collect();
    let ids: HashMap<_, _> = bm
        .iter()
        .enumerate()
        .map(|(i, m)| (m.id().to_owned(), i))
        .collect();
    ensure!(
        bm.len() == am.len() + plan["unions"].as_array().unwrap().len(),
        "Unexpected helper count"
    );
    ensure!(
        a.texcoord_buffer()
            == Some(&b.texcoord_buffer().unwrap()[..a.texcoord_buffer().unwrap().len()]),
        "Original UV changed"
    );
    let parameters = |m: &file::ParsedModel| {
        m.params()
            .into_iter()
            .map(|p| {
                (
                    p.id().to_owned(),
                    p.min().to_bits(),
                    p.max().to_bits(),
                    p.default().to_bits(),
                )
            })
            .collect::<Vec<_>>()
    };
    ensure!(
        parameters(&a) == parameters(&b),
        "Parameter contract changed"
    );
    let replacements: HashMap<Vec<String>, Vec<String>> = plan["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| {
            let strings = |v: &Value| {
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|s| s.as_str().unwrap().to_owned())
                    .collect()
            };
            (strings(&g["before"]), strings(&g["after"]))
        })
        .collect();
    for (x, y) in am.iter().zip(&bm) {
        ensure!(
            x.id() == y.id()
                && x.vertex_count() == y.vertex_count()
                && x.texture() == y.texture()
                && x.culling() == y.culling()
                && x.invert_mask() == y.invert_mask(),
            "Original mesh changed"
        );
        let clips: Vec<_> = x.clips().into_iter().map(|m| m.id().to_owned()).collect();
        let actual: Vec<_> = y.clips().into_iter().map(|m| m.id().to_owned()).collect();
        ensure!(
            actual == *replacements.get(&clips).unwrap_or(&clips),
            "Mask references do not match certified plan: {}",
            x.id()
        );
        let rx = x.index_range();
        let ry = y.index_range();
        ensure!(
            a.index_buffer().unwrap()[rx.start as usize..rx.end as usize]
                == b.index_buffer().unwrap()[ry.start as usize..ry.end as usize],
            "Original topology changed"
        );
    }
    let mut pairs = Vec::new();
    let mut leaves: HashMap<String, Vec<String>> = am
        .iter()
        .map(|m| (m.id().to_owned(), vec![m.id().to_owned()]))
        .collect();
    for u in plan["unions"].as_array().unwrap() {
        let name = u["id"].as_str().unwrap();
        let target = ids[name];
        let helper = &bm[target];
        ensure!(
            helper.clips().into_iter().next().is_none(),
            "Mask helper must be unclipped"
        );
        let mut offset = 0usize;
        let mut indices = Vec::new();
        let mut expanded = Vec::new();
        for source in u["sources"].as_array().unwrap() {
            let source = source.as_str().unwrap();
            let index = ids[source];
            let mesh = &bm[index];
            ensure!(mesh.culling() == helper.culling(), "Culling changed");
            let r = mesh.index_range();
            indices.extend(
                b.index_buffer().unwrap()[r.start as usize..r.end as usize]
                    .iter()
                    .map(|i| *i as usize + offset),
            );
            pairs.push((index, target, offset, mesh.vertex_count() as usize));
            offset += mesh.vertex_count() as usize;
            expanded.extend(leaves[source].clone());
        }
        ensure!(
            offset == helper.vertex_count() as usize,
            "Helper corner count changed"
        );
        let r = helper.index_range();
        ensure!(
            indices
                == b.index_buffer().unwrap()[r.start as usize..r.end as usize]
                    .iter()
                    .map(|i| *i as usize)
                    .collect::<Vec<_>>(),
            "Helper triangles changed"
        );
        leaves.insert(name.to_owned(), expanded);
    }
    for (before, after) in &replacements {
        let mut expected = before.clone();
        expected.sort();
        let mut actual = Vec::new();
        for source in after {
            actual.extend(leaves[source].clone());
        }
        actual.sort();
        ensure!(
            expected == actual,
            "Union expression changed original triangle multiplicities"
        );
    }
    let cases: Vec<Value> = serde_json::from_slice(&fs::read(&args[4])?)?;
    let mut da = Driver::new(&a);
    let mut db = Driver::new(&b);
    let mut rows = Vec::new();
    let mut total = 0usize;
    let mut bit_differences = 0usize;
    let mut helper_max = 0f32;
    for (i, c) in cases.iter().enumerate() {
        for p in a.params() {
            da.set_param(p.uid(), p.default())?;
        }
        for p in b.params() {
            db.set_param(p.uid(), p.default())?;
        }
        for (id, value) in c["parameters"].as_object().context("parameters")? {
            let v = value.as_f64().unwrap() as f32;
            da.set_param_by_id(id, v)?;
            db.set_param_by_id(id, v)?;
        }
        da.drive(&a);
        db.drive(&b);
        let mut different = 0;
        let mut error = 0f32;
        for (x, y) in am.iter().zip(&bm) {
            let x = da.artmesh_state(x.uid()).unwrap();
            let y = db.artmesh_state(y.uid()).unwrap();
            ensure!(
                x.depth == y.depth
                    && x.visual.visible == y.visual.visible
                    && x.visual.opacity == y.visual.opacity
                    && x.visual.multiply_color == y.visual.multiply_color
                    && x.visual.screen_color == y.visual.screen_color,
                "Original visual state changed"
            );
            for (v, w) in x.vertices.iter().zip(y.vertices) {
                ensure!(v.is_finite() && w.is_finite(), "Nonfinite vertex");
                different +=
                    usize::from(v.x.to_bits() != w.x.to_bits() || v.y.to_bits() != w.y.to_bits());
                error = error.max((*v - *w).length() * a.canvas_properties().scale);
                total += 1;
            }
        }
        for &(source, target, start, count) in &pairs {
            let s = db.artmesh_state(bm[source].uid()).unwrap();
            let t = db.artmesh_state(bm[target].uid()).unwrap();
            ensure!(
                s.visual.visible && t.visual.visible && t.visual.opacity == 0.,
                "Visibility assumptions violated"
            );
            for (v, w) in s.vertices.iter().zip(&t.vertices[start..start + count]) {
                ensure!(v.is_finite() && w.is_finite(), "Nonfinite helper corner");
                helper_max = helper_max.max((*v - *w).length() * a.canvas_properties().scale);
            }
        }
        bit_differences += different;
        rows.push(json!({"name":c["name"],"different_original_vertices":different,"max_original_px":error}));
        if i % 100 == 0 {
            println!(
                "Verified {}/{}; {} differences; helpers {} px",
                i + 1,
                cases.len(),
                bit_differences,
                helper_max
            );
        }
    }
    let report = json!({"poses":cases.len(),"original_meshes":am.len(),"added_helpers":bm.len()-am.len(),
        "compared_original_vertices":total,"different_original_vertices":bit_differences,
        "max_helper_copy_error_px":helper_max,"original_topology_uvs_parameters_visuals_checked":true,
        "mask_union_triangle_multiplicities_verified":true,"cases":rows});
    fs::write(&args[5], serde_json::to_vec(&report)?)?;
    ensure!(
        bit_differences == 0 && helper_max == 0.,
        "Mask batching changed native evaluation"
    );
    Ok(())
}
