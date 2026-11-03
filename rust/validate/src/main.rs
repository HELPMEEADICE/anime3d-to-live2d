// SPDX-License-Identifier: MIT
// External harness; upstream Ayagami remains unmodified. Uses its public parser, driver and renderer.
use anyhow::{Context, Result};
use ayagami::{
    core::{ArtMesh, Item, Model},
    file,
};
use ayagami_render::{
    ModelRenderer, RenderColorspace, RenderOptions,
    texture::{Texture, TextureManager},
};
use glam::{Affine2, uvec2, vec2};
use serde_json::{Value, json};
use std::{
    env, fs,
    fs::File,
    path::Path,
    sync::{Arc, mpsc},
    time::Instant,
};

fn main() -> Result<()> {
    pollster::block_on(run())
}
async fn run() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    anyhow::ensure!(
        args.len() == 4 || (args.len() == 5 && args[4] == "--images-only"),
        "Usage: popy-validate MODEL3.json CASES.json OUTPUT_DIR [--images-only]"
    );
    let images_only = args.len() == 5;
    let manifest: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let base = Path::new(&args[1]).parent().unwrap();
    let out = Path::new(&args[3]);
    fs::create_dir_all(out)?;
    let model = Arc::new(file::ParsedModel::load(&mut File::open(
        base.join(manifest["FileReferences"]["Moc"].as_str().unwrap()),
    )?)?);
    let inventory: Vec<Value> = model.artmeshes().into_iter().map(|mesh| {
        let range=mesh.index_range();
        json!({"id":mesh.id(),"vertices":mesh.vertex_count(),"culling":mesh.culling(),"invert_mask":mesh.invert_mask(),"texture":mesh.texture(),"clips":mesh.clips().into_iter().map(|m|m.id().to_string()).collect::<Vec<_>>(),"indices":model.index_buffer().unwrap()[range.start as usize..range.end as usize].to_vec()})
    }).collect();
    fs::write(
        out.join("model_inventory.json"),
        serde_json::to_vec(&inventory)?,
    )?;
    anyhow::ensure!(
        model.root_draw_group().is_some(),
        "MOC3 has no root draw group"
    );
    let canvas = model.canvas_properties();
    let (w, h) = (canvas.dimensions.x as u32, canvas.dimensions.y as u32);
    let texture_data: Vec<Vec<u8>> = manifest["FileReferences"]["Textures"]
        .as_array()
        .context("textures")?
        .iter()
        .map(|s| fs::read(base.join(s.as_str().unwrap())).unwrap())
        .collect();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        flags: Default::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await?;
    let info = adapter.get_info();
    println!("GPU {} {:?} canvas {}x{}", info.name, info.backend, w, h);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("POPY validation"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await?;
    let mut renderer = ModelRenderer::new(device.clone(), queue.clone())?;
    renderer.load_model(
        model.clone(),
        &texture_data
            .iter()
            .map(|b| b.as_slice())
            .collect::<Vec<_>>(),
    )?;
    let parameters = renderer.params();
    // Renderer internally flips Y. Runtime root points are (canvas-origin)/pixelsPerUnit.
    let transform = Affine2::from_scale(
        vec2(2.0 / canvas.dimensions.x, 2.0 / canvas.dimensions.y) * canvas.scale,
    );
    let options = RenderOptions {
        mask_dimensions: uvec2(w, h),
        colorspace: RenderColorspace::Linear,
        transform,
    };
    let cases: Vec<Value> = serde_json::from_slice(&fs::read(&args[2])?)?;
    let mut results = Vec::new();
    for case in cases {
        let name = case["name"].as_str().context("case name")?;
        let t = Instant::now();
        for p in &parameters {
            renderer.driver().set_param(p.uid, p.default)?;
        }
        for (id, v) in case["parameters"].as_object().context("case parameters")? {
            renderer
                .driver()
                .set_param_by_id(id, v.as_f64().unwrap() as f32)?;
        }
        let texture = Texture::new(
            &device,
            uvec2(w, h),
            wgpu::TextureFormat::Rgba8UnormSrgb,
            Some(1),
            Some("POPY offscreen"),
        );
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(name) });
        renderer.prepare(&mut encoder, &options);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(name),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            renderer.render(&mut pass, texture.texture.format());
        }
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely())?;
        let render_ms = t.elapsed().as_secs_f64() * 1000.;
        let geometry: Vec<Value> = if images_only {
            Vec::new()
        } else {
            model.artmeshes().into_iter().filter_map(|mesh|{
            renderer.driver().artmesh_state(mesh.uid()).map(|s|json!({"id":mesh.id(),"vertices":s.vertices.iter().map(|v|[v.x,v.y]).collect::<Vec<_>>(),"depth":s.depth,"visible":s.visual.visible,"opacity":s.visual.opacity,"multiply":[s.visual.multiply_color.x,s.visual.multiply_color.y,s.visual.multiply_color.z]}))
        }).collect()
        };
        // Preserve the original render for diagnosing GPU and alpha conversion independently.
        if name == "front" {
            let (tx, rx) = mpsc::channel();
            texture.download(&device, &queue, move |buf, bpr| {
                let _ = tx.send((buf.to_vec(), bpr));
            });
            device.poll(wgpu::PollType::wait_indefinitely())?;
            let (data, bpr) = rx.recv()?;
            let mut rgba = Vec::with_capacity((w * h * 4) as usize);
            for row in data.chunks(bpr as usize).take(h as usize) {
                rgba.extend_from_slice(&row[..(w * 4) as usize]);
            }
            image::save_buffer(
                out.join("front.premultiplied.png"),
                &rgba,
                w,
                h,
                image::ColorType::Rgba8,
            )?;
            println!(
                "CANVAS scale {} center {:?}, visible {}",
                canvas.scale,
                canvas.center,
                geometry.iter().filter(|v| v["visible"] == true).count()
            );
        }
        let texture = TextureManager::new(&device).unpremultiply(&device, &queue, &texture, true);
        let (tx, rx) = mpsc::channel();
        texture.download_to_image(&device, &queue, move |img| {
            let _ = tx.send(img);
        });
        device.poll(wgpu::PollType::wait_indefinitely())?;
        let img = rx.recv()?;
        img.save(out.join(format!("{name}.png")))?;
        anyhow::ensure!(
            img.to_rgba8().pixels().any(|p| p.0[3] > 128),
            "{name}: fully transparent rendering is a regression"
        );
        if !images_only {
            fs::write(
                out.join(format!("{name}.geometry.json")),
                serde_json::to_vec(&geometry)?,
            )?;
        }
        println!("RENDER {} {:.3} ms", name, render_ms);
        results.push(json!({"name":name,"pose_to_gpu_complete_ms":render_ms,"total_with_readback_and_files_ms":t.elapsed().as_secs_f64()*1000.}));
    }
    fs::write(
        out.join("runtime_report.json"),
        serde_json::to_vec_pretty(
            &json!({"ayagami_commit":"0d1d7aa3efe57b1b363b72e672e353c52a9c15d6","gpu":info.name,"backend":format!("{:?}",info.backend),"canvas":[w,h],"artmeshes":model.artmeshes().into_iter().count(),"parameters":parameters.iter().map(|p|json!({"id":p.id,"min":p.min,"max":p.max,"default":p.default})).collect::<Vec<_>>(),"cases":results}),
        )?,
    )?;
    Ok(())
}
