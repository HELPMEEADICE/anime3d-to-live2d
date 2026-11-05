// SPDX-License-Identifier: MIT
use anyhow::{Context, Result};
use ayagami::{
    core::{ArtMesh, BlendFormMap, Item, Model},
    file,
};
use serde_json::{Value, json};
use std::{env, fs, fs::File, path::Path};
fn main() -> Result<()> {
    let a: Vec<_> = env::args().collect();
    let p = Path::new(&a[1]);
    let m: Value = serde_json::from_slice(&fs::read(p)?)?;
    let model = file::ParsedModel::load(&mut File::open(
        p.parent()
            .unwrap()
            .join(m["FileReferences"]["Moc"].as_str().context("Moc")?),
    )?)?;
    let rows: Vec<_> = model
        .artmeshes()
        .into_iter()
        .map(|m| {
            let mut forms: Vec<_> = m.forms().into_iter().map(|f| f.uid()).collect();
            for b in m.blend_form_maps().into_iter().flatten() {
                forms.extend(b.forms().into_iter().map(|f| f.uid()));
            }
            json!({"id":m.id(),"vertices":m.vertex_count(),"forms":forms})
        })
        .collect();
    fs::write(&a[2], serde_json::to_vec(&json!({"meshes":rows}))?)?;
    Ok(())
}
