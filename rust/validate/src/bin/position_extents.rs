// SPDX-License-Identifier: MIT
// Record live position extents via the unchanged runtime's public model API.
use anyhow::{Context, Result, ensure};
use ayagami::{
    core::{
        ArtMesh, ArtMeshForm, BlendFormMap, Deformer, Item, Model, TypedDeformer, WarpDeformer,
        WarpForm,
    },
    file,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, env, fs, fs::File, path::Path};

fn record(map: &mut BTreeMap<u32, usize>, id: u32, length: usize) -> Result<()> {
    if let Some(old) = map.insert(id, length) {
        ensure!(old == length, "Form extent differs by owner");
    }
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    ensure!(args.len() == 3, "position_extents MODEL3.json OUTPUT.json");
    let path = Path::new(&args[1]);
    let manifest: Value = serde_json::from_slice(&fs::read(path)?)?;
    let moc = path
        .parent()
        .unwrap()
        .join(manifest["FileReferences"]["Moc"].as_str().context("Moc")?);
    let model = file::ParsedModel::load(&mut File::open(&moc)?)?;
    let mut art = BTreeMap::new();
    let mut warp = BTreeMap::new();
    for m in model.artmeshes() {
        for f in m.forms() {
            record(&mut art, f.uid(), 2 * f.vertices().len())?;
        }
        if let Some(blends) = m.blend_form_maps() {
            for b in blends {
                for f in b.forms() {
                    record(&mut art, f.uid(), 2 * f.vertices().len())?;
                }
            }
        }
    }
    for d in model.deformers() {
        if let TypedDeformer::Warp(w) = Deformer::typed(&*d) {
            for f in w.forms() {
                record(&mut warp, f.uid(), 2 * f.vertices().len())?;
            }
            if let Some(blends) = w.blend_form_maps() {
                for b in blends {
                    for f in b.forms() {
                        record(&mut warp, f.uid(), 2 * f.vertices().len())?;
                    }
                }
            }
        }
    }
    let dense = |m: BTreeMap<u32, usize>| -> Result<Vec<usize>> {
        ensure!(
            m.keys().copied().eq(0..m.len() as u32),
            "Unowned or noncontiguous form rows"
        );
        Ok(m.into_values().collect())
    };
    fs::write(
        &args[2],
        serde_json::to_vec(
            &json!({"moc":moc,"moc_bytes":fs::metadata(&moc)?.len(),"artmesh":dense(art)?,"warp":dense(warp)?}),
        )?,
    )?;
    Ok(())
}
