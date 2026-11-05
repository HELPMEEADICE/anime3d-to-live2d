// SPDX-License-Identifier: MIT
// Temporal test using the unmodified public Ayagami physics API.
use anyhow::{Result, ensure};
use ayagami::{
    core::{Model, Param},
    file, meta,
    physics::{PhysicsEngine, PhysicsOptions},
    pose::{Key, Pose},
};
use serde_json::{Value, json};
use std::{env, fs, fs::File, path::Path, time::Instant};

fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    ensure!(args.len() == 4, "physics_trace MODEL3 REPORT CASES");
    let path = Path::new(&args[1]);
    let base = path.parent().unwrap();
    let manifest: Value = serde_json::from_slice(&fs::read(path)?)?;
    let model = file::ParsedModel::load(&mut File::open(
        base.join(manifest["FileReferences"]["Moc"].as_str().unwrap()),
    )?)?;
    let config: meta::Physics3 = serde_json::from_slice(&fs::read(
        base.join(manifest["FileReferences"]["Physics"].as_str().unwrap()),
    )?)?;
    let outputs: Vec<String> = config
        .physics_settings
        .iter()
        .flat_map(|s| s.output.iter().map(|o| o.destination.id.clone()))
        .collect();
    let mut physics = PhysicsEngine::new(config, PhysicsOptions::compatible(None));
    let mut pose = Pose::new(&model);
    physics.settle(&pose);
    let mut samples = Vec::new();
    let mut cases = Vec::new();
    let mut micros = Vec::new();
    for frame in 0..1320 {
        let t = frame as f32 / 60.;
        for p in model.params() {
            pose.set(&Key::param(p.id()), p.default());
        }
        let mut input = serde_json::Map::new();
        if (1. ..5.).contains(&t) {
            let phase = (t - 1.) * std::f32::consts::TAU * 0.6;
            input.insert("ParamAngleX".into(), json!(35. * phase.sin()));
            input.insert("ParamAngleZ".into(), json!(15. * phase.sin()));
        }
        if (6. ..10.).contains(&t) {
            let phase = (t - 6.) * std::f32::consts::TAU * 0.7;
            input.insert("ParamBodyAngleX".into(), json!(10. * phase.sin()));
            input.insert("ParamBodyAngleZ".into(), json!(8. * phase.sin()));
            input.insert("ParamLegLSwing".into(), json!(15. * phase.sin()));
            input.insert("ParamLegRSwing".into(), json!(-15. * phase.sin()));
        }
        for (id, v) in &input {
            pose.set(&Key::param(id), v.as_f64().unwrap() as f32);
        }
        let now = Instant::now();
        physics.update(&mut pose, 1. / 60.);
        micros.push(now.elapsed().as_secs_f64() * 1e6);
        let values: Vec<f32> = outputs
            .iter()
            .map(|id| pose.get_flattened(&Key::param(id)).unwrap())
            .collect();
        ensure!(
            values.iter().all(|v| v.is_finite() && v.abs() <= 1.),
            "Unbounded physics"
        );
        samples.push(values.clone());
        // 10 fps preview of motion and first two seconds of settling.
        if frame % 6 == 0 && frame <= 720 {
            for (id, v) in outputs.iter().zip(&values) {
                input.insert(id.clone(), json!(v));
            }
            cases.push(json!({"name":format!("frame_{frame:04}"),"parameters":input}));
        }
    }
    let peaks: Vec<f32> = (0..outputs.len())
        .map(|i| samples.iter().map(|s| s[i].abs()).fold(0., f32::max))
        .collect();
    let tail: f32 = samples[1260..]
        .iter()
        .flatten()
        .map(|v| v.abs())
        .fold(0., f32::max);
    ensure!(peaks.iter().all(|v| *v > 0.01), "Disconnected output");
    ensure!(tail < 0.01, "Physics does not settle: {tail}");
    micros.sort_by(f64::total_cmp);
    fs::write(
        &args[2],
        serde_json::to_vec_pretty(
            &json!({"frames":1320,"fps":60,"seconds":22,"outputs":outputs,"peak_abs":peaks,"settled_last_second_max_abs":tail,"physics_only_median_us":micros[micros.len()/2],"physics_only_p95_us":micros[micros.len()*95/100],"runtime":"unmodified Ayagami compatible physics; not VTS", "samples":samples}),
        )?,
    )?;
    fs::write(&args[3], serde_json::to_vec(&cases)?)?;
    Ok(())
}
