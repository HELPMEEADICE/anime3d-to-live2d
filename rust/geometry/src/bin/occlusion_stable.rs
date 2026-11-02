// SPDX-License-Identifier: MIT
// Coverage-constrained orders, shared canonical layers and parameter-lattice smoothing.
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
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
    #[serde(default)]
    vertex_ranges: Vec<[usize; 2]>,
    #[serde(default)]
    neighbors: Vec<Vec<usize>>,
    #[serde(default)]
    priority_edges: Vec<[usize; 2]>,
}
#[derive(Clone, Copy)]
struct Edge {
    a: usize,
    b: usize,
    weight: u64,
}
#[derive(Default, Serialize)]
struct Diagnostics {
    poses: usize,
    removed_cycle_edges: u64,
    mutual_conflict_pixels: u64,
    conflict_meshes: Vec<u64>,
    stable_edges: usize,
    max_order: f32,
    max_constraint_violation: f32,
    smoothing_passes: usize,
}
fn raster(
    points: &[[f32; 3]],
    input: &Input,
    pixels: &mut [Vec<(usize, f32)>],
) -> (Vec<u32>, Vec<f32>) {
    let n = input.meshes.len();
    let (w, h) = (input.width, input.height);
    for p in pixels.iter_mut() {
        p.clear();
    }
    let mut baseline = vec![500.; n];
    for (mesh, tris) in input.meshes.iter().enumerate() {
        let mut total = 0.;
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
        if let Some(&[start, end]) = input.vertex_ranges.get(mesh) {
            baseline[mesh] = 500.
                - points[start..end].iter().map(|p| p[2]).sum::<f32>() / (end - start) as f32
                    * 800.;
        } else if !tris.is_empty() {
            baseline[mesh] = 500. - total / (tris.len() * 3) as f32 * 800.;
        }
    }
    let mut counts = vec![0u32; n * n];
    for p in pixels {
        if p.len() < 2 {
            continue;
        }
        p.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        let (winner, z) = p[0];
        let mut previous = usize::MAX;
        for &(other, depth) in p.iter().skip(1) {
            if other != winner && other != previous && depth - z > 1e-6 {
                counts[other * n + winner] += 1;
            }
            previous = other;
        }
    }
    (counts, baseline)
}
fn measured_edges(counts: &[u32], n: usize, diagnostics: &mut Diagnostics) -> Vec<Edge> {
    let mut edges = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            let (a, b) = (counts[i * n + j], counts[j * n + i]);
            if a >= 2 && b >= 2 {
                let conflict = a.min(b) as u64;
                diagnostics.mutual_conflict_pixels += conflict;
                diagnostics.conflict_meshes[i] += conflict;
                diagnostics.conflict_meshes[j] += conflict;
            }
            if a >= 2 && a > b {
                edges.push(Edge {
                    a: i,
                    b: j,
                    weight: a as u64,
                });
            } else if b >= 2 {
                edges.push(Edge {
                    a: j,
                    b: i,
                    weight: b as u64,
                });
            }
        }
    }
    edges
}
fn acyclic(edges: &[Edge], n: usize, diagnostics: &mut Diagnostics) -> Vec<Edge> {
    let mut outgoing = vec![Vec::new(); n];
    let mut incoming = vec![Vec::new(); n];
    let mut degree = vec![0usize; n];
    let mut alive = vec![true; edges.len()];
    let mut done = vec![false; n];
    for (i, e) in edges.iter().enumerate() {
        outgoing[e.a].push(i);
        incoming[e.b].push(i);
        degree[e.b] += 1;
    }
    let mut queue: VecDeque<_> = (0..n).filter(|&i| degree[i] == 0).collect();
    let mut topo = Vec::with_capacity(n);
    while topo.len() < n {
        if queue.is_empty() {
            let node = (0..n)
                .filter(|&i| !done[i])
                .min_by_key(|&i| {
                    incoming[i]
                        .iter()
                        .filter(|&&k| alive[k] && !done[edges[k].a])
                        .map(|&k| edges[k].weight)
                        .sum::<u64>()
                })
                .unwrap();
            for &k in &incoming[node] {
                if alive[k] && !done[edges[k].a] {
                    alive[k] = false;
                    diagnostics.removed_cycle_edges += 1;
                    let e = edges[k];
                    diagnostics.conflict_meshes[e.a] += e.weight.min(1 << 30);
                    diagnostics.conflict_meshes[e.b] += e.weight.min(1 << 30);
                }
            }
            degree[node] = 0;
            queue.push_back(node);
        }
        let i = queue.pop_front().unwrap();
        if done[i] {
            continue;
        }
        done[i] = true;
        topo.push(i);
        for &k in &outgoing[i] {
            if alive[k] {
                let j = edges[k].b;
                degree[j] -= 1;
                if degree[j] == 0 {
                    queue.push_back(j);
                }
            }
        }
    }
    let mut rank = vec![0; n];
    for (i, &node) in topo.iter().enumerate() {
        rank[node] = i;
    }
    let mut kept: Vec<_> = edges
        .iter()
        .enumerate()
        .filter(|(i, _)| alive[*i])
        .map(|(_, e)| *e)
        .collect();
    kept.sort_by_key(|e| (rank[e.a], rank[e.b]));
    kept
}
fn project_order(target: &[f32], edges: &[Edge]) -> Vec<f32> {
    // Dual coordinate descent / Dykstra. Both ends move: unlike upward-only propagation,
    // this minimizes L2 distance to the shared layer reference instead of accumulating drift.
    let mut value = target.to_vec();
    let mut lambda = vec![0f32; edges.len()];
    let margin = 2.0;
    for _ in 0..80 {
        let mut change = 0f32;
        for (i, e) in edges.iter().enumerate() {
            let next = (lambda[i] + (margin - value[e.b] + value[e.a]) * 0.5).max(0.);
            let delta = next - lambda[i];
            lambda[i] = next;
            value[e.a] -= delta;
            value[e.b] += delta;
            change = change.max(delta.abs());
        }
        if change < 0.002 {
            break;
        }
    }
    // Topologically ordered forward feasibility removes finite-iteration residuals.
    for e in edges {
        value[e.b] = value[e.b].max(value[e.a] + margin);
    }
    value
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 4 {
        return Err("Usage: occlusion_stable INPUT.json ORDERS.f32 DIAGNOSTICS.json".into());
    }
    let input: Input = serde_json::from_slice(&fs::read(&args[1])?)?;
    let n = input.meshes.len();
    let mut diagnostics = Diagnostics {
        poses: input.poses,
        conflict_meshes: vec![0; n],
        smoothing_passes: 3,
        ..Default::default()
    };
    let mut data = fs::File::open(&input.data)?;
    let mut bytes = vec![0u8; input.vertices * 12];
    let mut pixels = vec![Vec::new(); input.width * input.height];
    let mut all_edges = Vec::new();
    let mut baselines = Vec::new();
    let mut aggregate = vec![0u64; n * n];
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
        let (counts, baseline) = raster(&points, &input, &mut pixels);
        let edges = measured_edges(&counts, n, &mut diagnostics);
        for e in &edges {
            aggregate[e.a * n + e.b] += e.weight;
        }
        baselines.push(baseline);
        all_edges.push(edges);
        if pose % 60 == 0 {
            eprintln!("coverage {pose}/{}", input.poses);
        }
    }
    let mut stable: Vec<_> = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .filter_map(|(a, b)| {
            if aggregate[a * n + b] > 0 && aggregate[b * n + a] == 0 {
                Some(Edge {
                    a,
                    b,
                    weight: aggregate[a * n + b],
                })
            } else {
                None
            }
        })
        .collect();
    // Mask-protected feature layering is invariant. Do not let a rear skin patch or sclera
    // cover the pupil; the native depth-envelope masks handle actual foreground coverage.
    for &[a, b] in &input.priority_edges {
        stable.retain(|e| !(e.a == b && e.b == a) && !(e.a == a && e.b == b));
        stable.push(Edge {
            a,
            b,
            weight: 1 << 50,
        });
    }
    let stable = acyclic(&stable, n, &mut diagnostics);
    diagnostics.stable_edges = stable.len();
    let mean: Vec<_> = (0..n)
        .map(|i| baselines.iter().map(|b| b[i]).sum::<f32>() / input.poses as f32)
        .collect();
    let canonical = project_order(&mean, &stable);
    let mut constraints = Vec::new();
    for local in all_edges {
        let mut dense = vec![0u64; n * n];
        for e in local {
            dense[e.a * n + e.b] = e.weight * 1024;
        }
        for e in &stable {
            // Canonical relations outside the current silhouette are continuity hints,
            // not stronger evidence than actual depth at this pose. Hard global edges
            // can close a cycle through a changing local relation and discard true cover.
            let weight = if e.weight >= 1 << 50 { e.weight } else { 1 };
            dense[e.a * n + e.b] = dense[e.a * n + e.b].max(weight);
        }
        let union: Vec<_> = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .filter_map(|(a, b)| {
                if dense[a * n + b] > 0 {
                    Some(Edge {
                        a,
                        b,
                        weight: dense[a * n + b],
                    })
                } else {
                    None
                }
            })
            .collect();
        constraints.push(acyclic(&union, n, &mut diagnostics));
    }
    let mut orders: Vec<_> = baselines
        .iter()
        .zip(&constraints)
        .map(|(b, e)| {
            project_order(
                &(0..n)
                    .map(|i| canonical[i] + 0.10 * (b[i] - mean[i]))
                    .collect::<Vec<_>>(),
                e,
            )
        })
        .collect();
    for pass in 0..3 {
        let previous = orders.clone();
        for pose in 0..input.poses {
            let target: Vec<_> = (0..n)
                .map(|i| {
                    let adjacent = input
                        .neighbors
                        .get(pose)
                        .filter(|x| !x.is_empty())
                        .map(|neighbors| {
                            neighbors.iter().map(|&p| previous[p][i]).sum::<f32>()
                                / neighbors.len() as f32
                        })
                        .unwrap_or(canonical[i]);
                    0.55 * canonical[i] + 0.45 * adjacent + 0.05 * (baselines[pose][i] - mean[i])
                })
                .collect();
            orders[pose] = project_order(&target, &constraints[pose]);
        }
        eprintln!("order smoothing pass {}/3", pass + 1);
    }
    let mut out = fs::File::create(&args[2])?;
    let mut baseline_out = fs::File::create(format!("{}.baseline.f32", args[2]))?;
    for (pose, row) in orders.iter().enumerate() {
        for e in &constraints[pose] {
            diagnostics.max_constraint_violation = diagnostics
                .max_constraint_violation
                .max(2. - row[e.b] + row[e.a]);
        }
        for &v in row {
            diagnostics.max_order = diagnostics.max_order.max(v);
            out.write_all(&v.to_le_bytes())?;
        }
        for &v in &baselines[pose] {
            baseline_out.write_all(&v.to_le_bytes())?;
        }
    }
    fs::write(&args[3], serde_json::to_vec(&diagnostics)?)?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn minimal_symmetric_order_projection() {
        let p = project_order(
            &[10., 10.],
            &[Edge {
                a: 0,
                b: 1,
                weight: 1,
            }],
        );
        assert!((p[0] - 9.).abs() < 1e-4);
        assert!((p[1] - 11.).abs() < 1e-4);
    }
    #[test]
    fn cycles_are_recorded_and_ordered() {
        let e = vec![
            Edge {
                a: 0,
                b: 1,
                weight: 100,
            },
            Edge {
                a: 1,
                b: 2,
                weight: 100,
            },
            Edge {
                a: 2,
                b: 0,
                weight: 1,
            },
        ];
        let mut d = Diagnostics {
            conflict_meshes: vec![0; 3],
            ..Default::default()
        };
        let kept = acyclic(&e, 3, &mut d);
        assert_eq!(kept.len(), 2);
        assert_eq!(d.removed_cycle_edges, 1);
        assert!(d.conflict_meshes[0] > 0);
        let p = project_order(&[0.; 3], &kept);
        assert!(p[0] < p[1] && p[1] < p[2]);
    }
    #[test]
    fn mutual_surface_conflicts_are_not_hidden() {
        let mut d = Diagnostics {
            conflict_meshes: vec![0; 2],
            ..Default::default()
        };
        assert_eq!(measured_edges(&[0, 15, 5, 0], 2, &mut d).len(), 1);
        assert_eq!(d.mutual_conflict_pixels, 5);
        assert_eq!(d.conflict_meshes, vec![5, 5]);
    }
}
