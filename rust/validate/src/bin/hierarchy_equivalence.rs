// SPDX-License-Identifier: MIT
// Verify channel factoring through the unchanged public runtime.
use anyhow::{Context, Result, ensure};
use ayagami::{
    core::{ArtMesh, Glue, GlueForm, Item, ItemArray, Model, Param, ParamMap, Part},
    driver::{DrawNode, Driver},
    file,
};
use serde_json::{Value, json};
use std::{collections::HashMap, env, fs, fs::File, path::Path};
#[path = "../program_trace.rs"]
mod program_trace;

fn load(path: &Path) -> Result<file::ParsedModel> {
    let m: Value = serde_json::from_slice(&fs::read(path)?)?;
    Ok(file::ParsedModel::load(&mut File::open(
        path.parent()
            .unwrap()
            .join(m["FileReferences"]["Moc"].as_str().context("Moc")?),
    )?)?)
}
fn sequence(ids: &HashMap<u32, usize>, d: &Driver<file::ParsedModel>) -> Result<Vec<usize>> {
    let mut names = Vec::new();
    for node in d.draw_nodes(None).context("root nodes")? {
        match node {
            DrawNode::ArtMesh(uid) => {
                let s = d.artmesh_state(*uid).unwrap();
                if s.visual.visible && s.visual.opacity != 0. {
                    names.push(ids[uid]);
                }
            }
            DrawNode::OffscreenPart(_) => anyhow::bail!("Unexpected offscreen drawing"),
        }
    }
    Ok(names)
}
fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    ensure!(
        args.len() == 6,
        "hierarchy_equivalence BEFORE.model3.json AFTER.model3.json PLAN.json CASES.json REPORT.json"
    );
    let a = load(Path::new(&args[1]))?;
    let b = load(Path::new(&args[2]))?;
    let plan: Value = serde_json::from_slice(&fs::read(&args[3])?)?;
    let am: Vec<_> = a.artmeshes().into_iter().collect();
    let bm: Vec<_> = b.artmeshes().into_iter().collect();
    let ai = am.iter().enumerate().map(|(i, m)| (m.uid(), i)).collect();
    let bi = bm.iter().enumerate().map(|(i, m)| (m.uid(), i)).collect();
    let arena = plan.get("arena");
    let added = arena
        .map(|v| v["added_auxiliary_mesh_ids"].as_array().unwrap().len())
        .unwrap_or(0);
    ensure!(am.len() + added == bm.len(), "Unexpected mesh count");
    ensure!(
        a.texcoord_buffer()
            == Some(&b.texcoord_buffer().unwrap()[..a.texcoord_buffer().unwrap().len()]),
        "Original UV changed"
    );
    let params = |m: &file::ParsedModel| {
        m.params()
            .into_iter()
            .map(|p| {
                (
                    p.id().to_owned(),
                    p.min().to_bits(),
                    p.max().to_bits(),
                    p.default().to_bits(),
                )
            })
            .collect::<Vec<_>>()
    };
    ensure!(params(&a) == params(&b), "Parameter contract changed");
    for (x, y) in am.iter().zip(&bm) {
        let rx = x.index_range();
        let ry = y.index_range();
        ensure!(
            a.index_buffer().unwrap()[rx.start as usize..rx.end as usize]
                == b.index_buffer().unwrap()[ry.start as usize..ry.end as usize],
            "Original topology changed"
        );
        ensure!(
            x.id() == y.id()
                && x.vertex_count() == y.vertex_count()
                && x.texture() == y.texture()
                && x.culling() == y.culling()
                && x.invert_mask() == y.invert_mask(),
            "Mesh metadata changed"
        );
        ensure!(
            x.clips()
                .into_iter()
                .map(|m| m.id().to_owned())
                .collect::<Vec<_>>()
                == y.clips()
                    .into_iter()
                    .map(|m| m.id().to_owned())
                    .collect::<Vec<_>>(),
            "Mask sources changed"
        );
    }
    let trace_proof = if let Some(p) = arena {
        Some(program_trace::verify(&a, &b, p)?)
    } else {
        None
    };
    if arena.is_none() {
        ensure!(
            a.glues().count() == b.glues().count(),
            "Original Glue program changed"
        );
        let owners = |model: &file::ParsedModel| {
            model
                .params()
                .into_iter()
                .flat_map(|p| {
                    p.param_maps()
                        .into_iter()
                        .map(|m| (m.uid(), p.id().to_owned()))
                        .collect::<Vec<_>>()
                })
                .collect::<HashMap<_, _>>()
        };
        let ao = owners(&a);
        let bo = owners(&b);
        for (x, y) in a.glues().into_iter().zip(b.glues()) {
            ensure!(
                x.id() == y.id()
                    && x.artmesh_1().id() == y.artmesh_1().id()
                    && x.artmesh_2().id() == y.artmesh_2().id(),
                "Glue endpoints changed"
            );
            let maps = |g: &<file::ParsedModel as Model>::Glue<'_>,
                        owners: &HashMap<u32, String>| {
                g.param_maps()
                    .into_iter()
                    .map(|m| {
                        (
                            owners[&m.uid()].clone(),
                            m.keypoints()
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            ensure!(maps(&x, &ao) == maps(&y, &bo), "Glue axes changed");
            ensure!(
                x.forms()
                    .into_iter()
                    .map(|f| f.compatibility().to_bits())
                    .collect::<Vec<_>>()
                    == y.forms()
                        .into_iter()
                        .map(|f| f.compatibility().to_bits())
                        .collect::<Vec<_>>(),
                "Glue field changed"
            );
            let coords = |g: &<file::ParsedModel as Model>::Glue<'_>| {
                g.coords()
                    .into_iter()
                    .map(|p| {
                        [
                            (p[0].vertex_index, p[0].weight.to_bits()),
                            (p[1].vertex_index, p[1].weight.to_bits()),
                        ]
                    })
                    .collect::<Vec<_>>()
            };
            ensure!(
                coords(&x) == coords(&y),
                "Glue pair order or weights changed"
            );
        }
    }
    let part_ids: HashMap<_, _> = b
        .parts()
        .into_iter()
        .map(|p| (p.id().to_owned(), p.uid()))
        .collect();
    let mut owner = HashMap::new();
    let a_part_ids: HashMap<_, _> = a
        .parts()
        .into_iter()
        .map(|p| (p.id().to_owned(), p.uid()))
        .collect();
    let mut a_owner = HashMap::new();
    for g in plan["groups"].as_array().context("groups")? {
        for name in g["mesh_ids"].as_array().unwrap() {
            owner.insert(
                name.as_str().unwrap().to_owned(),
                part_ids[g["id"].as_str().unwrap()],
            );
            if let Some(uid) = a_part_ids.get(g["id"].as_str().unwrap()) {
                a_owner.insert(name.as_str().unwrap().to_owned(), *uid);
            }
        }
    }
    let cases: Vec<Value> = serde_json::from_slice(&fs::read(&args[4])?)?;
    let (mut da, mut db) = (Driver::new(&a), Driver::new(&b));
    let (mut vertices, mut different, mut visuals, mut orders, mut sequences) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut max_error, mut max_visual) = (0f32, 0f32);
    let mut rows = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        for p in a.params() {
            da.set_param(p.uid(), p.default())?;
        }
        for p in b.params() {
            db.set_param(p.uid(), p.default())?;
        }
        for (id, v) in c["parameters"].as_object().unwrap() {
            da.set_param_by_id(id, v.as_f64().unwrap() as f32)?;
            db.set_param_by_id(id, v.as_f64().unwrap() as f32)?;
        }
        da.drive(&a);
        db.drive(&b);
        let (mut dv, mut dc, mut dd) = (0, 0, 0);
        for (x, y) in am.iter().zip(&bm) {
            let sx = da.artmesh_state(x.uid()).unwrap();
            let sy = db.artmesh_state(y.uid()).unwrap();
            let effective_depth = owner
                .get(x.id())
                .map(|uid| db.part_state(*uid).unwrap().depth)
                .unwrap_or(sy.depth);
            let a_depth = a_owner
                .get(x.id())
                .map(|uid| da.part_state(*uid).unwrap().depth)
                .unwrap_or(sx.depth);
            dd += usize::from(a_depth != effective_depth);
            ensure!(sx.visual.visible == sy.visual.visible, "Visibility changed");
            let vx = [
                sx.visual.opacity,
                sx.visual.multiply_color.x,
                sx.visual.multiply_color.y,
                sx.visual.multiply_color.z,
                sx.visual.screen_color.x,
                sx.visual.screen_color.y,
                sx.visual.screen_color.z,
            ];
            let vy = [
                sy.visual.opacity,
                sy.visual.multiply_color.x,
                sy.visual.multiply_color.y,
                sy.visual.multiply_color.z,
                sy.visual.screen_color.x,
                sy.visual.screen_color.y,
                sy.visual.screen_color.z,
            ];
            for (v, w) in vx.into_iter().zip(vy) {
                dc += usize::from(v.to_bits() != w.to_bits());
                max_visual = max_visual.max((v - w).abs());
            }
            for (v, w) in sx.vertices.iter().zip(sy.vertices) {
                ensure!(v.is_finite() && w.is_finite(), "Nonfinite vertex");
                dv += usize::from(v.x.to_bits() != w.x.to_bits() || v.y.to_bits() != w.y.to_bits());
                max_error = max_error.max((*v - *w).length() * a.canvas_properties().scale);
                vertices += 1;
            }
        }
        for m in bm.iter().skip(am.len()) {
            let state = db.artmesh_state(m.uid()).unwrap();
            ensure!(
                state.visual.opacity == 0. && m.clips().into_iter().next().is_none(),
                "Auxiliary mesh became rendered"
            );
        }
        let seq = sequence(&ai, &da)? != sequence(&bi, &db)?;
        different += dv;
        visuals += dc;
        orders += dd;
        sequences += usize::from(seq);
        rows.push(json!({"name":c["name"],"different_vertices":dv,"different_visual_values":dc,"different_effective_depths":dd,"different_render_sequence":seq}));
        if i % 100 == 0 {
            println!(
                "{}/{}: vertices {} visuals {} depths {} sequences {}; max {} px",
                i + 1,
                cases.len(),
                different,
                visuals,
                orders,
                sequences,
                max_error
            );
        }
    }
    let report = json!({"poses":cases.len(),"compared_vertices":vertices,"different_vertices":different,"max_vertex_error_px":max_error,"different_visual_values":visuals,"max_visual_error":max_visual,"different_effective_depths":orders,"different_render_sequences":sequences,"original_glue_fields_pairs_and_order_identical":arena.is_none(),"program_trace_proof":trace_proof,"cases":rows});
    fs::write(&args[5], serde_json::to_vec(&report)?)?;
    ensure!(
        different == 0 && visuals == 0 && orders == 0 && sequences == 0,
        "Factoring changed evaluation; inspect report"
    );
    Ok(())
}
