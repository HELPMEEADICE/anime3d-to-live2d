// SPDX-License-Identifier: MIT
// Exact vertex/surface correspondence after adjacent ArtMesh concatenation.
use anyhow::{Context, Result, ensure};
use ayagami::{
    core::{ArtMesh, Item, Model, Param},
    driver::Driver,
    file,
};
use serde_json::{Value, json};
use std::{collections::HashMap, env, fs, fs::File, path::Path};

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
        "batch_equivalence BEFORE AFTER MAPPING CASES REPORT"
    );
    let before = load(Path::new(&args[1]))?;
    let after = load(Path::new(&args[2]))?;
    let am: Vec<_> = before.artmeshes().into_iter().collect();
    let bm: Vec<_> = after.artmeshes().into_iter().collect();
    let aid: HashMap<_, _> = am.iter().map(|m| (m.id(), m)).collect();
    let bid: HashMap<_, _> = bm.iter().map(|m| (m.id(), m)).collect();
    let mapping: Vec<Value> = serde_json::from_slice(&fs::read(&args[3])?)?;
    ensure!(mapping.len() == am.len(), "incomplete mapping");
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
    for row in &mapping {
        let a = aid[row["source"].as_str().unwrap()];
        let b = bid[row["target"].as_str().unwrap()];
        let vs = row["vertex_start"].as_u64().unwrap() as usize;
        let n = a.vertex_count() as usize;
        ensure!(
            a.texture() == b.texture()
                && a.culling() == b.culling()
                && a.invert_mask() == b.invert_mask(),
            "flags changed"
        );
        ensure!(
            a.clips()
                .into_iter()
                .map(|m| m.id().to_owned())
                .collect::<Vec<_>>()
                == b.clips()
                    .into_iter()
                    .map(|m| m.id().to_owned())
                    .collect::<Vec<_>>(),
            "clipping changed"
        );
        let au = a.texcoord_offset() as usize;
        let bu = b.texcoord_offset() as usize + vs;
        ensure!(
            before.texcoord_buffer().unwrap()[au..au + n]
                == after.texcoord_buffer().unwrap()[bu..bu + n],
            "UV changed"
        );
        let ar = a.index_range();
        let br = b.index_range();
        let bi = br.start as usize + row["index_start"].as_u64().unwrap() as usize;
        for (&x, &y) in before.index_buffer().unwrap()[ar.start as usize..ar.end as usize]
            .iter()
            .zip(&after.index_buffer().unwrap()[bi..bi + (ar.end - ar.start) as usize])
        {
            ensure!(x as usize + vs == y as usize, "triangle changed");
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
        for row in &mapping {
            let x = aid[row["source"].as_str().unwrap()];
            let y = bid[row["target"].as_str().unwrap()];
            let sx = a.artmesh_state(x.uid()).unwrap();
            let sy = b.artmesh_state(y.uid()).unwrap();
            let start = row["vertex_start"].as_u64().unwrap() as usize;
            ensure!(
                sx.depth == sy.depth
                    && sx.visual.visible == sy.visual.visible
                    && sx.visual.opacity.to_bits() == sy.visual.opacity.to_bits()
                    && sx.visual.multiply_color == sy.visual.multiply_color
                    && sx.visual.screen_color == sy.visual.screen_color,
                "visual changed {} {}",
                i,
                x.id()
            );
            for (vx, vy) in sx
                .vertices
                .iter()
                .zip(&sy.vertices[start..start + sx.vertices.len()])
            {
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
            &json!({"poses":cases.len(),"compared_vertices":count,"different_vertices":0,"before_meshes":am.len(),"after_meshes":bm.len(),"topology_uv_masks_visuals_equal":true,"method":"Unmodified public driver; every source mesh corresponds to its exact contiguous vertex/index slice in the adjacent batch. GPU order/alpha checked separately."}),
        )?,
    )?;
    Ok(())
}
