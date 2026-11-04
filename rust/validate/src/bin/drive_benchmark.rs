// SPDX-License-Identifier: MIT
// Paired public-Driver timing; no renderer/GPU or upstream changes.
use anyhow::{Context, Result, ensure};
use ayagami::{
    core::{Item, Model, Param},
    driver::Driver,
    file,
};
use serde_json::{Value, json};
use std::{env, fs, fs::File, path::Path, time::Instant};

struct Candidate {
    label: String,
    model: file::ParsedModel,
    driver: Driver<file::ParsedModel>,
    defaults: Vec<(u32, f32)>,
    probe: u32,
}
fn pose(c: &mut Candidate, case: &Value) -> Result<f64> {
    for &(id, value) in &c.defaults {
        c.driver.set_param(id, value)?;
    }
    for (id, value) in case["parameters"].as_object().context("parameters")? {
        c.driver
            .set_param_by_id(id, value.as_f64().context("parameter value")? as f32)?;
    }
    let t = Instant::now();
    c.driver.drive(&c.model);
    let ms = t.elapsed().as_secs_f64() * 1000.;
    std::hint::black_box(c.driver.artmesh_state(c.probe).context("probe")?.vertices);
    Ok(ms)
}
fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    ensure!(
        args.len() == 4,
        "drive_benchmark MODELS.json CASES.json OUTPUT.json"
    );
    let config: Vec<Value> = serde_json::from_slice(&fs::read(&args[1])?)?;
    let cases: Vec<Value> = serde_json::from_slice(&fs::read(&args[2])?)?;
    ensure!(
        cases.len() > 5 && config.len() >= 2,
        "Need warmup, timed cases and at least two candidates"
    );
    let mut candidates = Vec::new();
    for entry in &config {
        let path = Path::new(entry["manifest"].as_str().context("manifest")?);
        let manifest: Value = serde_json::from_slice(&fs::read(path)?)?;
        let model = file::ParsedModel::load(&mut File::open(
            path.parent()
                .unwrap()
                .join(manifest["FileReferences"]["Moc"].as_str().context("Moc")?),
        )?)?;
        let defaults = model
            .params()
            .into_iter()
            .map(|p| (p.uid(), p.default()))
            .collect();
        let probe = model.artmeshes().into_iter().next().context("mesh")?.uid();
        let driver = Driver::new(&model);
        candidates.push(Candidate {
            label: entry["label"].as_str().context("label")?.to_owned(),
            model,
            driver,
            defaults,
            probe,
        });
    }
    for c in &mut candidates {
        for case in &cases[..5] {
            pose(c, case)?;
        }
    }
    let mut rows = Vec::new();
    // Every model follows the same 60-pose sequence; only inter-model order
    // rotates/reverses. Both loaded data and driver caches stay resident.
    for round in 0..4 {
        for (i, case) in cases[5..].iter().enumerate() {
            for position in 0..candidates.len() {
                let k = if round % 2 == 0 {
                    position
                } else {
                    candidates.len() - 1 - position
                };
                let index = (k + i + round) % candidates.len();
                let c = &mut candidates[index];
                let ms = pose(c, case)?;
                rows.push(json!({"round":round,"case":case["name"],"position":position,"label":c.label,"driver_ms":ms}));
            }
        }
        eprintln!("Completed paired CPU round {}", round + 1);
    }
    fs::write(
        &args[3],
        serde_json::to_vec(
            &json!({"models":config,"warmup_per_model":5,"rounds":4,"timed_poses_per_round":cases.len()-5,"samples":rows,"note":"Same-process public Ayagami Driver only, rotating/reversed per-pose model order; no GPU, textures, buffer upload, or frame-time claim. Desktop not isolated."}),
        )?,
    )?;
    Ok(())
}
