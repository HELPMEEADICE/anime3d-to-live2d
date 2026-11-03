//! Find visible source-depth / ArtMesh-order conflicts without image-derived topology.
use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::{collections::BTreeMap, env, fs, path::Path};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    anyhow::ensure!(args.len() == 3, "face_conflicts INPUT.json OUTPUT.json");
    let input: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let base = Path::new(&args[1]).parent().context("input parent")?;
    let w = input["width"].as_u64().context("width")? as usize;
    let h = input["height"].as_u64().context("height")? as usize;
    let meshes = input["meshes"].as_array().context("meshes")?;
    let mut scores = vec![[0u64; 2]; meshes.len()];
    let mut triangle_scores: BTreeMap<(usize, usize), u64> = BTreeMap::new();
    let mut pose_counts = Vec::new();
    for case in input["cases"].as_array().context("cases")? {
        let data = fs::read(base.join(case["file"].as_str().context("file")?))?;
        anyhow::ensure!(data.len() % 48 == 0, "bad triangle records");
        let mut actual_order = vec![f32::NEG_INFINITY; w * h];
        let mut depth = vec![f32::INFINITY; w * h];
        let mut actual = vec![usize::MAX; w * h];
        let mut truth = actual.clone();
        let mut actual_triangle = vec![0usize; w * h];
        for record in data.chunks_exact(48) {
            let mut a = [0f32; 12];
            for (i, word) in record.chunks_exact(4).enumerate() {
                a[i] = f32::from_le_bytes(word.try_into()?);
            }
            let p = [[a[0], a[1], a[2]], [a[3], a[4], a[5]], [a[6], a[7], a[8]]];
            let index = a[10] as usize;
            anyhow::ensure!(index < meshes.len(), "mesh index");
            let u = [p[1][0] - p[0][0], p[1][1] - p[0][1]];
            let v = [p[2][0] - p[0][0], p[2][1] - p[0][1]];
            let det = u[0] * v[1] - u[1] * v[0];
            if det <= 1e-8 {
                continue;
            }
            let x0 = p
                .iter()
                .map(|q| q[0])
                .fold(f32::INFINITY, f32::min)
                .floor()
                .max(0.) as usize;
            let x1 = p
                .iter()
                .map(|q| q[0])
                .fold(f32::NEG_INFINITY, f32::max)
                .ceil()
                .max(0.)
                .min(w as f32) as usize;
            let y0 = p
                .iter()
                .map(|q| q[1])
                .fold(f32::INFINITY, f32::min)
                .floor()
                .max(0.) as usize;
            let y1 = p
                .iter()
                .map(|q| q[1])
                .fold(f32::NEG_INFINITY, f32::max)
                .ceil()
                .max(0.)
                .min(h as f32) as usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    let q = [x as f32 + 0.5 - p[0][0], y as f32 + 0.5 - p[0][1]];
                    let s = (q[0] * v[1] - q[1] * v[0]) / det;
                    let t = (u[0] * q[1] - u[1] * q[0]) / det;
                    if s < 0. || t < 0. || s + t > 1. {
                        continue;
                    }
                    let pixel = y * w + x;
                    if a[9].floor() >= actual_order[pixel] {
                        actual_order[pixel] = a[9].floor();
                        actual[pixel] = index;
                        actual_triangle[pixel] = a[11] as usize;
                    }
                    let z = p[0][2] * (1. - s - t) + p[1][2] * s + p[2][2] * t;
                    if z < depth[pixel] {
                        depth[pixel] = z;
                        truth[pixel] = index;
                    }
                }
            }
        }
        let mut total = 0u64;
        for (pixel, (&a, &b)) in actual.iter().zip(&truth).enumerate() {
            if a == usize::MAX || b == usize::MAX || a == b {
                continue;
            }
            let ak = meshes[a]["kind"].as_u64().context("kind")?;
            let bk = meshes[b]["kind"].as_u64().context("kind")?;
            if ak == bk {
                continue;
            }
            scores[a][0] += 1;
            scores[b][1] += 1;
            *triangle_scores
                .entry((a, actual_triangle[pixel]))
                .or_default() += 1;
            total += 1;
        }
        pose_counts.push(json!({"name": case["name"], "conflict_pixels": total}));
    }
    let ranked: Vec<_> = meshes.iter().enumerate().filter(|(i, _)| scores[*i] != [0, 0])
        .map(|(i, m)| json!({"mesh":m["id"],"wrong_cover":scores[i][0],"wrong_covered":scores[i][1]})).collect();
    let triangles: Vec<_> = triangle_scores
        .iter()
        .map(|(&(i, t), &count)| json!({"mesh":meshes[i]["id"], "triangle":t, "wrong_cover":count}))
        .collect();
    fs::write(
        &args[2],
        serde_json::to_vec(
            &json!({"meshes":ranked,"triangles":triangles,"cases":pose_counts,
        "method":"Original source triangles rasterized with barycentric camera depth versus authored per-ArtMesh order, front-face culling, full source samples. Screening only; final acceptance requires independent Blender/native MOC renders."}),
        )?,
    )?;
    Ok(())
}
