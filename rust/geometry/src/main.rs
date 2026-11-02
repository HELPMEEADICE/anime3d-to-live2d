// SPDX-License-Identifier: MIT
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    env, fs,
};
#[derive(Deserialize)]
struct Input {
    triangles: Vec<[usize; 3]>,
    groups: Vec<u64>,
    limit: usize,
    #[serde(default)]
    limits: Vec<usize>,
}
#[derive(Serialize)]
struct Output {
    patches: Vec<Vec<usize>>,
}
fn partition(input: &Input) -> Output {
    let mut incident: HashMap<usize, Vec<usize>> = HashMap::new();
    for (t, tri) in input.triangles.iter().enumerate() {
        for &v in tri {
            incident.entry(v).or_default().push(t);
        }
    }
    let mut used = vec![false; input.triangles.len()];
    let mut patches = Vec::new();
    for start in 0..used.len() {
        if used[start] {
            continue;
        }
        let mut patch = Vec::new();
        let mut q = VecDeque::from([start]);
        while let Some(t) = q.pop_front() {
            if used[t] || input.groups[t] != input.groups[start] {
                continue;
            }
            let limit = input.limits.get(start).copied().unwrap_or(input.limit);
            if patch.len() >= limit {
                break;
            }
            used[t] = true;
            patch.push(t);
            for v in input.triangles[t] {
                for &n in &incident[&v] {
                    if !used[n] {
                        q.push_back(n)
                    }
                }
            }
        }
        patches.push(patch);
    }
    Output { patches }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = env::args().collect();
    if a.len() != 3 {
        return Err("Usage: popy-geometry INPUT.json OUTPUT.json".into());
    }
    let input: Input = serde_json::from_slice(&fs::read(&a[1])?)?;
    if input.groups.len() != input.triangles.len() || input.limit == 0 {
        return Err("invalid groups/limit".into());
    }
    let result = partition(&input);
    fs::write(&a[2], serde_json::to_vec(&result)?)?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn connectivity_semantics_and_budget() {
        let i = Input {
            triangles: vec![[0, 1, 2], [1, 2, 3], [2, 3, 4], [10, 11, 12]],
            groups: vec![1, 1, 2, 1],
            limit: 1,
            limits: Vec::new(),
        };
        let p = partition(&i);
        let mut all = p.patches.iter().flatten().copied().collect::<Vec<_>>();
        all.sort();
        assert_eq!(all, vec![0, 1, 2, 3]);
        assert_eq!(p.patches.len(), 4);
        let i = Input { limit: 80, ..i };
        let p = partition(&i);
        assert_eq!(p.patches, vec![vec![0, 1], vec![2], vec![3]]);
    }
    #[test]
    fn adaptive_limit_preserves_all_source_triangles() {
        let i = Input {
            triangles: vec![[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]],
            groups: vec![0; 4],
            limit: 48,
            limits: vec![2; 4],
        };
        let p = partition(&i);
        assert_eq!(p.patches, vec![vec![0, 1], vec![2, 3]]);
    }
}
