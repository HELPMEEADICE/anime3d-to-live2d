// SPDX-License-Identifier: MIT
// External test harness using the unchanged, pinned Ayagami public driver API.
use anyhow::{Context, Result};
use ayagami::{
    core::{ArtMesh, Item, ItemArray, Model, Param},
    driver::Driver,
    file,
};
use serde_json::{Value, json};
use std::{
    env,
    fs::{self, File},
    path::Path,
    time::Instant,
};

fn load(path: &Path) -> Result<file::ParsedModel> {
    let manifest: Value = serde_json::from_slice(&fs::read(path)?)?;
    Ok(file::ParsedModel::load(&mut File::open(
        path.parent()
            .unwrap()
            .join(manifest["FileReferences"]["Moc"].as_str().context("Moc")?),
    )?)?)
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    anyhow::ensure!(
        args.len() == 5 || (args.len() == 6 && args[5] == "--allow-difference"),
        "equivalence BEFORE.model3.json AFTER.model3.json CASES.json REPORT.json [--allow-difference]"
    );
    let before = load(Path::new(&args[1]))?;
    let after = load(Path::new(&args[2]))?;
    let mut a = Driver::new(&before);
    let mut b = Driver::new(&after);
    let am: Vec<_> = before.artmeshes().into_iter().collect();
    let bm: Vec<_> = after.artmeshes().into_iter().collect();
    anyhow::ensure!(
        before.texcoord_buffer() == after.texcoord_buffer(),
        "UV changed"
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
                    p.repeat(),
                    p.is_blendshape(),
                )
            })
            .collect::<Vec<_>>()
    };
    anyhow::ensure!(
        parameters(&before) == parameters(&after),
        "parameter contract changed"
    );
    anyhow::ensure!(am.len() == bm.len(), "mesh count changed");
    for (x, y) in am.iter().zip(&bm) {
        anyhow::ensure!(
            x.id() == y.id()
                && x.vertex_count() == y.vertex_count()
                && x.culling() == y.culling()
                && x.invert_mask() == y.invert_mask()
                && x.texture() == y.texture(),
            "mesh identity/flags changed"
        );
        anyhow::ensure!(
            x.clips()
                .into_iter()
                .map(|v| v.id().to_owned())
                .collect::<Vec<_>>()
                == y.clips()
                    .into_iter()
                    .map(|v| v.id().to_owned())
                    .collect::<Vec<_>>(),
            "mask changed"
        );
        let rx = x.index_range();
        let ry = y.index_range();
        anyhow::ensure!(
            before.index_buffer().unwrap()[rx.start as usize..rx.end as usize]
                == after.index_buffer().unwrap()[ry.start as usize..ry.end as usize],
            "triangles changed"
        );
    }
    let cases: Vec<Value> = serde_json::from_slice(&fs::read(&args[3])?)?;
    let mut rows = Vec::new();
    let mut global_max = 0.0f32;
    let mut total_differences = 0usize;
    let mut total_vertices = 0usize;
    let scale = before.canvas_properties().scale;
    for (index, case) in cases.iter().enumerate() {
        for p in before.params() {
            a.set_param(p.uid(), p.default())?;
        }
        for p in after.params() {
            b.set_param(p.uid(), p.default())?;
        }
        for (id, v) in case["parameters"].as_object().context("parameters")? {
            a.set_param_by_id(id, v.as_f64().unwrap() as f32)?;
            b.set_param_by_id(id, v.as_f64().unwrap() as f32)?;
        }
        let start = Instant::now();
        a.drive(&before);
        let a_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        b.drive(&after);
        let b_ms = start.elapsed().as_secs_f64() * 1000.;
        let mut max = 0.0f32;
        let mut differences = 0usize;
        let mut visible_max = 0.0f32;
        let mut face_max = 0.0f32;
        let mut worst = String::new();
        for (x, y) in am.iter().zip(&bm) {
            let sx = a.artmesh_state(x.uid()).context("before state")?;
            let sy = b.artmesh_state(y.uid()).context("after state")?;
            anyhow::ensure!(
                sx.depth == sy.depth
                    && sx.visual.visible == sy.visual.visible
                    && sx.visual.opacity == sy.visual.opacity
                    && sx.visual.multiply_color == sy.visual.multiply_color
                    && sx.visual.screen_color == sy.visual.screen_color,
                "visual/depth changed at {} {}",
                case["name"],
                x.id()
            );
            for (vx, vy) in sx.vertices.iter().zip(sy.vertices) {
                anyhow::ensure!(vx.is_finite() && vy.is_finite(), "nonfinite vertex");
                let error = (*vx - *vy).length() * scale;
                if error > max {
                    max = error;
                    worst = x.id().to_owned();
                }
                if sx.visual.opacity > 0. && sx.visual.visible {
                    visible_max = visible_max.max(error);
                }
                if ["M_Face_", "M_Eye_", "M_Brow_", "M_Hair_", "M_Headwear_"]
                    .iter()
                    .any(|s| x.id().starts_with(s))
                {
                    face_max = face_max.max(error);
                }
                differences += usize::from(
                    vx.x.to_bits() != vy.x.to_bits() || vx.y.to_bits() != vy.y.to_bits(),
                );
                total_vertices += 1;
            }
        }
        global_max = global_max.max(max);
        total_differences += differences;
        rows.push(json!({"name":case["name"],"max_px_including_hidden":max,
            "visible_max_px":visible_max,"face_hair_headwear_max_px":face_max,"different_vertices":differences,"worst_mesh":worst,
            "before_driver_ms":a_ms,"after_driver_ms":b_ms}));
        if index % 100 == 0 {
            println!(
                "Compared {}/{}; max {} px",
                index + 1,
                cases.len(),
                global_max
            );
        }
    }
    let report = json!({"poses":cases.len(),"compared_vertices":total_vertices,
        "artmeshes_before":am.len(),"artmeshes_after":bm.len(),
        "glues_before":before.glues().count(),"glues_after":after.glues().count(),
        "different_vertices":total_differences,"max_px_including_hidden":global_max,
        "topology_masks_flags_and_evaluated_visuals_equal":true,
        "method":"Two unmodified pinned Ayagami drivers, identical parameter sequence, every vertex including hidden carriers; timings interleaved before/after, CPU only", "cases":rows});
    fs::write(&args[4], serde_json::to_vec(&report)?)?;
    println!(
        "{} poses, {} different vertices, max {} px",
        cases.len(),
        total_differences,
        global_max
    );
    if args.len() == 5 {
        anyhow::ensure!(total_differences == 0, "optimization is not bit-exact");
    }
    Ok(())
}
