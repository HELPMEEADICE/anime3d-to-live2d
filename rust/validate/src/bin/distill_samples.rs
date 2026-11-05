// SPDX-License-Identifier: MIT
// Offline teacher sampling through the unchanged Ayagami public driver.
use anyhow::{Context, Result, ensure};
use ayagami::{
    core::{ArtMesh, Item, Model, Param},
    driver::Driver,
    file,
};
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    env, fs,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    ensure!(args.len() == 4, "distill_samples MODEL3 CASES OUTPUT_DIR");
    let path = Path::new(&args[1]);
    let out = Path::new(&args[3]);
    fs::create_dir_all(out)?;
    let manifest: Value = serde_json::from_slice(&fs::read(path)?)?;
    let model = file::ParsedModel::load(&mut File::open(
        path.parent()
            .unwrap()
            .join(manifest["FileReferences"]["Moc"].as_str().context("Moc")?),
    )?)?;
    let all: Vec<_> = model.artmeshes().into_iter().collect();
    let sources: HashSet<_> = all
        .iter()
        .flat_map(|m| m.clips().into_iter().map(|s| s.uid()))
        .collect();
    // Full_* are the actual visible full-body surfaces, not native matrix
    // carriers. Keep them alongside the face/hair meshes and clipping sources.
    let meshes: Vec<_> = all
        .iter()
        .filter(|m| {
            m.id().starts_with("M_") || m.id().starts_with("Full_") || sources.contains(&m.uid())
        })
        .collect();
    let indices = model.index_buffer().context("indices")?;
    let uv = model.texcoord_buffer().context("uv")?;
    let canvas = model.canvas_properties();
    let mut cursor = 0usize;
    let metadata: Vec<_> = meshes
        .iter()
        .map(|m| {
            let n = m.vertex_count() as usize;
            let start = cursor;
            cursor += n;
            let range = m.index_range();
            let tex = m.texcoord_offset() as usize;
            json!({"id":m.id(),"start":start,"vertices":n,"texture":m.texture(),
            "culling":m.culling(),"invert_mask":m.invert_mask(),"visible":m.visible(),
            "clips":m.clips().into_iter().map(|s|s.id().to_string()).collect::<Vec<_>>(),
            "uvs":uv[tex..tex+n].iter().flat_map(|v|[v.x,v.y]).collect::<Vec<_>>(),
            "indices":indices[range.start as usize..range.end as usize].to_vec()})
        })
        .collect();
    let parameters: Vec<_> = model
        .params()
        .into_iter()
        .map(|p| json!({"id":p.id(),"min":p.min(),"max":p.max(),"default":p.default()}))
        .collect();
    fs::write(
        out.join("metadata.json"),
        serde_json::to_vec(&json!({"meshes":metadata,"parameters":parameters,
        "canvas":[canvas.dimensions.x,canvas.dimensions.y],"vertices":cursor,
        "center":[canvas.center.x,canvas.center.y],"scale":canvas.scale,"format_version":2}))?,
    )?;
    let cases: Vec<Value> = serde_json::from_slice(&fs::read(&args[2])?)?;
    let mut positions = BufWriter::new(File::create(out.join("positions.f32"))?);
    let mut channels = BufWriter::new(File::create(out.join("channels.f32"))?);
    let mut driver = Driver::new(&model);
    for (i, case) in cases.iter().enumerate() {
        for p in model.params() {
            driver.set_param(p.uid(), p.default())?;
        }
        for (id, v) in case["parameters"].as_object().context("parameters")? {
            driver.set_param_by_id(id, v.as_f64().context("value")? as f32)?;
        }
        driver.drive(&model);
        for m in &meshes {
            let state = driver.artmesh_state(m.uid()).context("state")?;
            for v in state.vertices {
                // Public root coordinates -> source canvas pixels (Y down).
                positions.write_all(&(v.x * canvas.scale + canvas.center.x).to_le_bytes())?;
                positions.write_all(&(canvas.center.y + v.y * canvas.scale).to_le_bytes())?;
            }
            channels.write_all(&(state.depth as f32).to_le_bytes())?;
            channels.write_all(
                &(if state.visual.visible {
                    state.visual.opacity
                } else {
                    0.
                })
                .to_le_bytes(),
            )?;
        }
        if i % 25 == 0 {
            eprintln!("{}/{} teacher poses", i + 1, cases.len());
        }
    }
    positions.flush()?;
    channels.flush()?;
    Ok(())
}
