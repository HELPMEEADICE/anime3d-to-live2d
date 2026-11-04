// SPDX-License-Identifier: MIT
// Inspect actual MOC data through the pinned runtime's public read-only API.
use anyhow::{Context, Result, ensure};
use ayagami::{
    core::{ArtMesh, Deformer, Glue, Item, ItemArray, Model},
    file,
};
use serde_json::{Value, json};
use std::{collections::HashMap, env, fs, fs::File, path::Path};

fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    ensure!(args.len() == 3, "model_stats MODEL3.json OUTPUT.json");
    let manifest_path = Path::new(&args[1]);
    let base = manifest_path.parent().unwrap();
    let manifest: Value = serde_json::from_slice(&fs::read(manifest_path)?)?;
    let refs = &manifest["FileReferences"];
    let moc = base.join(refs["Moc"].as_str().context("Moc")?);
    let model = file::ParsedModel::load(&mut File::open(&moc)?)?;
    let meshes: Vec<_> = model.artmeshes().into_iter().collect();
    let mut max_depth = 0;
    let parents: HashMap<_, _> = model
        .deformers()
        .into_iter()
        .map(|d| (d.uid(), d.parent().map(|p| p.uid())))
        .collect();
    for id in parents.keys() {
        let mut depth = 1;
        let mut parent = parents[id];
        while let Some(node) = parent {
            depth += 1;
            ensure!(depth < 1024, "Cyclic deformer graph");
            parent = parents[&node];
        }
        max_depth = max_depth.max(depth);
    }
    let mut textures = Vec::new();
    for p in refs["Textures"].as_array().context("Textures")? {
        let (w, h) = image::image_dimensions(base.join(p.as_str().unwrap()))?;
        textures.push(json!({"width":w,"height":h}));
    }
    let result = json!({
        "moc_bytes":fs::metadata(moc)?.len(),"includes_hidden_carriers":true,
        "artmeshes":meshes.len(),
        "vertices":meshes.iter().map(|m|m.vertex_count() as usize).sum::<usize>(),
        "triangles":meshes.iter().map(|m|{let r=m.index_range();(r.end-r.start) as usize/3}).sum::<usize>(),
        "artmesh_keyforms":meshes.iter().map(|m|m.forms().count()).sum::<usize>(),
        "glues":model.glues().count(),
        "glue_keyforms":model.glues().into_iter().map(|g|g.forms().count()).sum::<usize>(),
        "deformers":model.deformers().count(),"max_deformer_depth":max_depth,
        "parameters":model.params().count(),"textures":textures,
        "motions":refs["Motions"].as_object().map(|groups|groups.values().map(|g|g.as_array().unwrap().len()).sum::<usize>()).unwrap_or(0),
        "expressions":refs["Expressions"].as_array().map(|v|v.len()).unwrap_or(0),
    });
    fs::write(&args[2], serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}
