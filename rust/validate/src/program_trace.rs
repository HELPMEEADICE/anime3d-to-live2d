// SPDX-License-Identifier: MIT
// Independent proof of ordered per-vertex arithmetic after register relocation.
use anyhow::{Context, Result, ensure};
use ayagami::{
    core::{ArtMesh, Glue, GlueForm, Item, Model, Param, ParamMap},
    file,
};
use serde_json::{Value, json};
use std::collections::HashMap;

#[derive(Hash, PartialEq, Eq)]
struct Field {
    axes: Vec<(String, Vec<u32>)>,
    values: Vec<u32>,
}
#[derive(Hash, PartialEq, Eq)]
struct Event {
    a: usize,
    b: usize,
    va: u32,
    vb: u32,
    wa: u32,
    wb: u32,
    field: usize,
    side: u8,
}

pub fn verify(a: &file::ParsedModel, b: &file::ParsedModel, plan: &Value) -> Result<Value> {
    let am: Vec<_> = a.artmeshes().into_iter().collect();
    let bm: Vec<_> = b.artmeshes().into_iter().collect();
    let old_ids: HashMap<_, _> = am
        .iter()
        .enumerate()
        .map(|(i, m)| (m.id().to_owned(), i))
        .collect();
    let new_ids: HashMap<_, _> = bm
        .iter()
        .enumerate()
        .map(|(i, m)| (m.id().to_owned(), i))
        .collect();
    let mut bases = vec![0usize; am.len() + 1];
    for (i, m) in am.iter().enumerate() {
        bases[i + 1] = bases[i] + m.vertex_count() as usize;
    }
    let mut logical: Vec<Vec<usize>> = bm
        .iter()
        .enumerate()
        .map(|(i, m)| {
            if i < am.len() {
                (bases[i]..bases[i + 1]).collect()
            } else {
                vec![usize::MAX; m.vertex_count() as usize]
            }
        })
        .collect();
    let mut first = vec![None; bases[am.len()]];
    let auid: HashMap<_, _> = am.iter().enumerate().map(|(i, m)| (m.uid(), i)).collect();
    for g in a.glues() {
        let ma = auid[&g.artmesh_1().uid()];
        let mb = auid[&g.artmesh_2().uid()];
        let one = g.param_maps().into_iter().next().is_none()
            && g.forms().into_iter().all(|f| f.compatibility() == 1.);
        for [x, y] in g.coords() {
            let ia = bases[ma] + x.vertex_index as usize;
            let ib = bases[mb] + y.vertex_index as usize;
            first[ia].get_or_insert(one && x.weight == 1. && y.weight == 0. && ia != ib);
            first[ib].get_or_insert(one && y.weight == 1. && x.weight == 0. && ia != ib);
        }
    }
    let mapping = plan["mapping"].as_object().context("arena mapping")?;
    for (name, m) in mapping {
        let old = old_ids[name];
        let arena = new_ids[m["arena"].as_str().context("arena id")?];
        let start = m["start"].as_u64().unwrap() as usize;
        let count = m["vertices"].as_u64().unwrap() as usize;
        ensure!(
            count == am[old].vertex_count() as usize && start + count <= logical[arena].len(),
            "Arena range mismatch"
        );
        for i in 0..count {
            ensure!(
                first[bases[old] + i] == Some(true),
                "Initial value is read before overwrite: {name}:{i}"
            );
            ensure!(
                logical[arena][start + i] == usize::MAX,
                "Overlapping registers"
            );
            logical[arena][start + i] = bases[old] + i;
        }
    }
    for m in &logical {
        ensure!(
            m.iter().all(|x| *x != usize::MAX),
            "Unmapped auxiliary register"
        );
    }
    let mut fields: HashMap<Field, usize> = HashMap::new();
    let mut events: HashMap<Event, usize> = HashMap::new();
    let mut results = Vec::new();
    let mut total_pairs = Vec::new();
    for (pass, model) in [a, b].into_iter().enumerate() {
        let meshes: Vec<_> = model.artmeshes().into_iter().collect();
        let ids: HashMap<_, _> = meshes
            .iter()
            .enumerate()
            .map(|(i, m)| (m.uid(), i))
            .collect();
        let owners: HashMap<_, _> = model
            .params()
            .into_iter()
            .flat_map(|p| {
                p.param_maps()
                    .into_iter()
                    .map(|m| (m.uid(), p.id().to_owned()))
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut versions: Vec<Vec<u32>> = meshes
            .iter()
            .map(|m| vec![0; m.vertex_count() as usize])
            .collect();
        let mut traces = vec![Vec::new(); bases[am.len()]];
        let mut copies = vec![false; bases[am.len()]];
        let mut pairs = 0;
        for g in model.glues() {
            let ma = ids[&g.artmesh_1().uid()];
            let mb = ids[&g.artmesh_2().uid()];
            let axes: Vec<_> = g
                .param_maps()
                .into_iter()
                .map(|m| {
                    (
                        owners[&m.uid()].clone(),
                        m.keypoints().iter().map(|v| v.to_bits()).collect(),
                    )
                })
                .collect();
            let values: Vec<_> = g
                .forms()
                .into_iter()
                .map(|f| f.compatibility().to_bits())
                .collect();
            let one = axes.is_empty() && values == vec![1f32.to_bits()];
            let key = Field { axes, values };
            let next = fields.len();
            let field = *fields.entry(key).or_insert(next);
            let output_copy = pass == 1 && g.id().starts_with("G_ArenaOutput_");
            for [x, y] in g.coords() {
                pairs += 1;
                let ia = x.vertex_index as usize;
                let ib = y.vertex_index as usize;
                let la = if pass == 0 {
                    bases[ma] + ia
                } else {
                    logical[ma][ia]
                };
                let lb = if pass == 0 {
                    bases[mb] + ib
                } else {
                    logical[mb][ib]
                };
                let va = versions[ma][ia];
                let vb = versions[mb][ib];
                if output_copy {
                    ensure!(
                        one && x.weight == 0.
                            && y.weight == 1.
                            && la == lb
                            && ma >= am.len()
                            && mb < am.len(),
                        "Invalid final copy"
                    );
                    ensure!(!copies[lb], "Repeated output copy");
                    copies[lb] = true;
                    versions[mb][ib] = va;
                    continue;
                }
                for (side, write) in [(0, !(one && x.weight == 0.)), (1, !(one && y.weight == 0.))]
                {
                    if !write {
                        continue;
                    }
                    let event = Event {
                        a: la,
                        b: lb,
                        va,
                        vb,
                        wa: x.weight.to_bits(),
                        wb: y.weight.to_bits(),
                        field,
                        side,
                    };
                    let next = events.len();
                    let token = *events.entry(event).or_insert(next);
                    let target = if side == 0 { la } else { lb };
                    traces[target].push(token);
                    if side == 0 {
                        versions[ma][ia] += 1;
                    } else {
                        versions[mb][ib] += 1;
                    }
                }
            }
        }
        if pass == 1 {
            for name in mapping.keys() {
                let i = old_ids[name];
                ensure!(
                    copies[bases[i]..bases[i + 1]].iter().all(|x| *x),
                    "Missing output copy"
                );
            }
        }
        results.push((traces, versions));
        total_pairs.push(pairs);
    }
    ensure!(
        results[0].0 == results[1].0,
        "Read versions / ordered write programs changed"
    );
    for i in 0..am.len() {
        ensure!(
            results[0].1[i] == results[1].1[i],
            "Final output version mismatch"
        );
    }
    Ok(
        json!({"original_vertices":bases[am.len()],"original_pairs":total_pairs[0],"lowered_pairs":total_pairs[1],
        "initial_values_proven_dead":true,"per_vertex_write_order_and_source_versions_identical":true,
        "scalar_fields_compared_by_parameter_keys_and_float_bits":true,"final_output_copies_verified":true,
        "canonical_endpoint_identity_rule":"compatibility one and side weight zero are readonly; actual finite float results separately tested"}),
    )
}
