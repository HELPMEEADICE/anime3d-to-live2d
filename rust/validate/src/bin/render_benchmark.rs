// SPDX-License-Identifier: MIT
// Same-device paired frame timing through the unchanged public renderer.
use anyhow::{Context, Result, ensure};
use ayagami::{core::Model, file};
use ayagami_render::{ModelRenderer, RenderColorspace, RenderOptions, texture::Texture};
use glam::{Affine2, uvec2, vec2};
use serde_json::{Value, json};
use std::{env, fs, fs::File, path::Path, sync::Arc, time::Instant};

type Renderer = ModelRenderer<file::ParsedModel, Arc<file::ParsedModel>>;
fn frame(
    r: &mut Renderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target: &Texture,
    options: &RenderOptions,
    case: &Value,
) -> Result<Value> {
    let t = Instant::now();
    for p in r.params() {
        r.driver().set_param(p.uid, p.default)?;
    }
    for (id, value) in case["parameters"].as_object().context("parameters")? {
        r.driver()
            .set_param_by_id(id, value.as_f64().context("value")? as f32)?;
    }
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let prepare = Instant::now();
    r.prepare(&mut encoder, options);
    let prepare_ms = prepare.elapsed().as_secs_f64() * 1000.;
    let encode = Instant::now();
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        r.render(&mut pass, target.texture.format());
    }
    let encoded = encoder.finish();
    let encode_ms = encode.elapsed().as_secs_f64() * 1000.;
    let wait = Instant::now();
    queue.submit([encoded]);
    device.poll(wgpu::PollType::wait_indefinitely())?;
    Ok(
        json!({"pose_to_gpu_complete_ms":t.elapsed().as_secs_f64()*1000.,"prepare_ms":prepare_ms,"encode_ms":encode_ms,"gpu_submit_and_wait_ms":wait.elapsed().as_secs_f64()*1000.}),
    )
}
async fn execute() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    ensure!(
        args.len() == 5,
        "render_benchmark BEFORE.model3.json AFTER.model3.json CASES.json OUTPUT.json"
    );
    let cases: Vec<Value> = serde_json::from_slice(&fs::read(&args[3])?)?;
    ensure!(cases.len() > 5, "Need warmup and timed poses");
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::DX12,
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
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;
    let mut renderers = Vec::new();
    let mut canvas = None;
    for path in &args[1..3] {
        let path = Path::new(path);
        let m: Value = serde_json::from_slice(&fs::read(path)?)?;
        let base = path.parent().unwrap();
        let model = Arc::new(file::ParsedModel::load(&mut File::open(
            base.join(m["FileReferences"]["Moc"].as_str().context("Moc")?),
        )?)?);
        let c = model.canvas_properties();
        let size = (
            c.dimensions.x as u32,
            c.dimensions.y as u32,
            c.scale.to_bits(),
        );
        if let Some(previous) = canvas {
            ensure!(previous == size, "Canvas mismatch");
        } else {
            canvas = Some(size);
        }
        let textures: Vec<Vec<u8>> = m["FileReferences"]["Textures"]
            .as_array()
            .context("Textures")?
            .iter()
            .map(|p| fs::read(base.join(p.as_str().unwrap())))
            .collect::<std::io::Result<_>>()?;
        let mut r = ModelRenderer::new(device.clone(), queue.clone())?;
        r.load_model(
            model,
            &textures.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        )?;
        renderers.push(r);
    }
    let (w, h, scale) = canvas.unwrap();
    let options = RenderOptions {
        mask_dimensions: uvec2(w, h),
        colorspace: RenderColorspace::Linear,
        transform: Affine2::from_scale(vec2(2. / w as f32, 2. / h as f32) * f32::from_bits(scale)),
    };
    let target = Texture::new(
        &device,
        uvec2(w, h),
        wgpu::TextureFormat::Rgba8UnormSrgb,
        Some(1),
        Some("Paired target"),
    );
    for r in &mut renderers {
        for c in &cases[..5] {
            frame(r, &device, &queue, &target, &options, c)?;
        }
    }
    let mut rows = Vec::new();
    for round in 0..4 {
        for (i, c) in cases[5..].iter().enumerate() {
            for position in 0..2 {
                let index = (position + i + round) % 2;
                let mut row = frame(&mut renderers[index], &device, &queue, &target, &options, c)?;
                row["label"] = json!(if index == 0 { "baseline" } else { "candidate" });
                row["case"] = c["name"].clone();
                row["round"] = json!(round);
                row["position"] = json!(position);
                rows.push(row);
            }
        }
        eprintln!("Completed paired DX12 round {}", round + 1);
    }
    fs::write(
        &args[4],
        serde_json::to_vec(
            &json!({"gpu":info.name,"backend":"Dx12","rounds":4,"warmup_per_model":5,"samples":rows,"note":"Same-device alternating poses, both models resident, same full-resolution target/masks, GPU completion included, no readback. Relative diagnostic; two-model residency differs from a single-model host. Desktop not isolated."}),
        )?,
    )?;
    Ok(())
}
fn main() -> Result<()> {
    pollster::block_on(execute())
}
