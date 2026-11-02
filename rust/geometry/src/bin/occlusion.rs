// SPDX-License-Identifier: MIT
// Build painter constraints from 3D depth at overlapping projected pixels, instead of center depth.
use serde::{Deserialize, Serialize};
use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    env, fs,
    io::{Read, Write},
};
#[derive(Deserialize)]
struct Input {
    meshes: Vec<Vec<[usize; 3]>>,
    vertices: usize,
    poses: usize,
    width: usize,
    height: usize,
    data: String,
}
#[derive(Serialize)]
struct Diagnostics {
    poses: usize,
    removed_cycle_edges: u64,
    max_order: f32,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = env::args().collect();
    if a.len() != 4 {
        return Err("Usage: occlusion INPUT.json ORDERS.f32 DIAGNOSTICS.json".into());
    }
    let input: Input = serde_json::from_slice(&fs::read(&a[1])?)?;
    let n = input.meshes.len();
    let (w, h) = (input.width, input.height);
    let mut data = fs::File::open(&input.data)?;
    let mut out = fs::File::create(&a[2])?;
    let mut bytes = vec![0u8; input.vertices * 12];
    let mut pixels: Vec<Vec<(usize, f32)>> = vec![Vec::new(); w * h];
    let mut removed = 0u64;
    let mut max_order = 0f32;
    for pose in 0..input.poses {
        data.read_exact(&mut bytes)?;
        let points: Vec<[f32; 3]> = bytes
            .chunks_exact(12)
            .map(|b| {
                [
                    f32::from_le_bytes(b[0..4].try_into().unwrap()),
                    f32::from_le_bytes(b[4..8].try_into().unwrap()),
                    f32::from_le_bytes(b[8..12].try_into().unwrap()),
                ]
            })
            .collect();
        for p in &mut pixels {
            p.clear()
        }
        let mut baseline = vec![500f32; n];
        for (mesh, tris) in input.meshes.iter().enumerate() {
            let mut total = 0f32;
            for t in tris {
                let [aa, bb, cc] = [points[t[0]], points[t[1]], points[t[2]]];
                total += aa[2] + bb[2] + cc[2];
                let (ax, ay, bx, by, cx, cy) = (
                    aa[0] * w as f32,
                    aa[1] * h as f32,
                    bb[0] * w as f32,
                    bb[1] * h as f32,
                    cc[0] * w as f32,
                    cc[1] * h as f32,
                );
                let area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
                if area <= 1e-7 {
                    continue;
                }
                let (x0, x1) = (
                    ax.min(bx).min(cx).floor().max(0.) as usize,
                    ax.max(bx).max(cx).ceil().max(0.).min(w as f32) as usize,
                );
                let (y0, y1) = (
                    ay.min(by).min(cy).floor().max(0.) as usize,
                    ay.max(by).max(cy).ceil().max(0.).min(h as f32) as usize,
                );
                for y in y0..y1 {
                    for x in x0..x1 {
                        let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                        let u = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area;
                        let v = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area;
                        let z = 1. - u - v;
                        if u >= 0. && v >= 0. && z >= 0. {
                            pixels[y * w + x].push((mesh, aa[2] * u + bb[2] * v + cc[2] * z));
                        }
                    }
                }
            }
            if !tris.is_empty() {
                baseline[mesh] = 500. - total / (tris.len() * 3) as f32 * 800.;
            }
        }
        let mut edges = vec![0u32; n * n];
        for p in &mut pixels {
            if p.len() < 2 {
                continue;
            }
            p.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            let winner = p[0].0;
            for &(other, depth) in p.iter().skip(1) {
                if other != winner && depth - p[0].1 > 1e-6 {
                    edges[other * n + winner] += 1;
                }
            }
        }
        // Ignore single-pixel numerical boundary contacts. Preserve stronger evidence in mutual conflicts.
        for i in 0..n {
            for j in i + 1..n {
                let (a, b) = (edges[i * n + j], edges[j * n + i]);
                if a < 2 || a <= b {
                    edges[i * n + j] = 0;
                }
                if b < 2 || b < a {
                    edges[j * n + i] = 0;
                }
            }
        }
        let mut indegree = vec![0usize; n];
        for i in 0..n {
            for j in 0..n {
                if edges[i * n + j] > 0 {
                    indegree[j] += 1;
                }
            }
        }
        let mut q = BinaryHeap::new();
        let mut done = vec![false; n];
        let mut order = baseline.clone();
        for i in 0..n {
            if indegree[i] == 0 {
                q.push(Reverse((baseline[i] as i32, i)));
            }
        }
        for _ in 0..n {
            if q.is_empty() {
                let node = (0..n)
                    .filter(|&i| !done[i])
                    .min_by_key(|&i| {
                        (0..n)
                            .filter(|&j| !done[j])
                            .map(|j| edges[j * n + i] as u64)
                            .sum::<u64>()
                    })
                    .unwrap();
                for j in 0..n {
                    if !done[j] && edges[j * n + node] > 0 {
                        edges[j * n + node] = 0;
                        removed += 1;
                    }
                }
                indegree[node] = 0;
                q.push(Reverse((baseline[node] as i32, node)));
            }
            let Reverse((_, i)) = q.pop().unwrap();
            done[i] = true;
            for j in 0..n {
                if edges[i * n + j] > 0 {
                    order[j] = order[j].max(order[i].ceil() + 1.);
                    indegree[j] -= 1;
                    if indegree[j] == 0 {
                        q.push(Reverse((baseline[j] as i32, j)));
                    }
                }
            }
        }
        for value in order {
            max_order = max_order.max(value);
            out.write_all(&value.to_le_bytes())?;
        }
        if pose % 30 == 0 {
            eprintln!("occlusion {pose}/{}", input.poses);
        }
    }
    fs::write(
        &a[3],
        serde_json::to_vec(&Diagnostics {
            poses: input.poses,
            removed_cycle_edges: removed,
            max_order,
        })?,
    )?;
    Ok(())
}
