use serde_json::{Value, json};
use std::{
    fs,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn scan(hair_order: f32, reverse: bool) -> Value {
    let dir = std::env::temp_dir().join(format!(
        "popy-depth-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&dir).unwrap();
    let mut records = vec![0f32, 0., -1., 4., 0., -1., 0., 4., -1., 2., 0., 10.];
    let hair = if reverse {
        [0., 0., 0., 0., 4., 0., 4., 0., 0.]
    } else {
        [0., 0., 0., 4., 0., 0., 0., 4., 0.]
    };
    records.extend(hair);
    records.extend([hair_order, 1., 20.]);
    fs::write(
        dir.join("tri.f32"),
        records
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    fs::write(dir.join("input.json"), serde_json::to_vec(&json!({"width":4,"height":4,"meshes":[{"id":"skin","kind":0},{"id":"hair","kind":1}],"cases":[{"name":"pair","file":"tri.f32"}]})).unwrap()).unwrap();
    assert!(
        Command::new(env!("CARGO_BIN_EXE_face_conflicts"))
            .arg(dir.join("input.json"))
            .arg(dir.join("out.json"))
            .status()
            .unwrap()
            .success()
    );
    let result = serde_json::from_slice(&fs::read(dir.join("out.json")).unwrap()).unwrap();
    for file in ["input.json", "tri.f32", "out.json"] {
        fs::remove_file(dir.join(file)).unwrap();
    }
    fs::remove_dir(dir).unwrap();
    result
}

#[test]
fn detects_true_depth_mismatch_and_original_triangle() {
    let result = scan(3., false);
    assert_eq!(result["cases"][0]["conflict_pixels"], 10);
    assert!(
        result["triangles"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["mesh"] == "hair" && r["triangle"] == 20 && r["wrong_cover"] == 10)
    );
}

#[test]
fn correct_depth_and_culled_triangle_have_no_conflicts() {
    assert_eq!(scan(1., false)["cases"][0]["conflict_pixels"], 0);
    assert_eq!(scan(3., true)["cases"][0]["conflict_pixels"], 0);
}
