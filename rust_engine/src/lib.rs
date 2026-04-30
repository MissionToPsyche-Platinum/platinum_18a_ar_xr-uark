//! Asteroid collision and rover movement engine.
//! Exposes WASM functions for mesh loading, surface raycasting, and rover positioning.

use nalgebra::{Isometry3, Point3, Vector3};
use parry3d::query::{Ray, RayCast};
use parry3d::shape::TriMesh;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use wasm_bindgen::prelude::*;
use web_sys::XrSessionMode;

#[derive(Serialize, Deserialize)]
pub struct RoverTransform {
    pub position: [f32; 3],
}

#[derive(Serialize, Deserialize)]
pub struct WaypointPlacement {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

static ASTEROID_MESH: Mutex<Option<TriMesh>> = Mutex::new(None);

/// Requests WebXR immersive AR session for marker-based tracking.
#[wasm_bindgen]
pub async fn start_ar_session(mode: String) -> Result<(), JsValue> {
    let window = web_sys::window().ok_or("no window")?;
    let navigator = window.navigator();
    let xr_system = navigator.xr();

    let _session =
        wasm_bindgen_futures::JsFuture::from(xr_system.request_session(XrSessionMode::ImmersiveAr))
            .await?;

    web_sys::console::log_1(&format!("Started AR session in {} mode", mode).into());
    Ok(())
}

/// Loads GLB, merges all meshes into one TriMesh, applies scale and offset to match visual model.
#[wasm_bindgen]
pub async fn load_collision_mesh(glb_data: Vec<u8>) -> Result<(), JsValue> {
    web_sys::console::log_1(&"Loading collision mesh...".into());

    let (gltf, buffers, _images) = gltf::import_slice(&glb_data)
        .map_err(|e| JsValue::from_str(&format!("GLTF parse error: {:?}", e)))?;

    let scale_factor: f32 = 2.5;
    let offset = Vector3::new(-3.75_f32, -2.2, 3.22); /* Must match A-Frame model position. */

    let mut all_vertices: Vec<Point3<f32>> = Vec::new();
    let mut all_indices: Vec<[u32; 3]> = Vec::new();

    for mesh in gltf.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let positions = match reader.read_positions() {
                Some(p) => p,
                None => continue,
            };

            let base_vertex = all_vertices.len() as u32;

            all_vertices.extend(positions.map(|p| {
                Point3::new(
                    p[0] * scale_factor + offset.x,
                    p[1] * scale_factor + offset.y,
                    p[2] * scale_factor + offset.z,
                )
            }));

            if let Some(indices) = reader.read_indices() {
                let idx: Vec<u32> = indices.into_u32().collect();
                all_indices.extend(idx.chunks(3).filter_map(|c| {
                    if c.len() == 3 {
                        Some([c[0] + base_vertex, c[1] + base_vertex, c[2] + base_vertex])
                    } else {
                        None
                    }
                }));
            }
        }
    }

    if all_vertices.is_empty() || all_indices.is_empty() {
        return Err(JsValue::from_str("No geometry found in GLB"));
    }

    web_sys::console::log_1(
        &format!(
            "✅ Collision mesh: {} verts, {} tris (scaled {}×, offset {:?})",
            all_vertices.len(),
            all_indices.len(),
            scale_factor,
            offset.as_slice()
        )
        .into(),
    );

    let trimesh = TriMesh::new(all_vertices, all_indices);
    *ASTEROID_MESH.lock().unwrap() = Some(trimesh);
    Ok(())
}

/// Projects direction onto tangent plane, moves rover, raycasts to surface, returns new position.
#[wasm_bindgen]
pub fn move_rover_on_asteroid(
    dir_x: f32,
    dir_y: f32,
    dir_z: f32,
    current_x: f32,
    current_y: f32,
    current_z: f32,
) -> Result<JsValue, JsValue> {
    let mesh_guard = ASTEROID_MESH.lock().unwrap();
    let mesh = mesh_guard
        .as_ref()
        .ok_or(JsValue::from_str("Collision mesh not loaded yet"))?;

    let rover_pos = Point3::new(current_x, current_y, current_z);
    let speed: f32 = 0.08;

    let surface_normal = (rover_pos - Point3::origin()).normalize();

    /* Project direction onto tangent plane at rover position. */
    let desired = Vector3::new(dir_x, dir_y, dir_z);
    let projected = desired - surface_normal * desired.dot(&surface_normal);

    let projected_len = projected.norm();

    let movement = if projected_len > 0.001 {
        projected * (speed / projected_len.max(1.0))
    } else {
        Vector3::zeros()
    };
    let desired_pos = rover_pos + movement;

    /* Raycast from outside toward center to find surface intersection. */
    let to_center = Point3::origin() - desired_pos;
    let ray_dir = to_center.normalize();
    let ray = Ray::new(desired_pos - ray_dir * 15.0, ray_dir);

    if let Some(t) = mesh.cast_ray(&Isometry3::identity(), &ray, 30.0, true) {
        let hit = ray.origin + ray.dir * t;
        let hover = 0.08;
        let final_pos = hit - ray_dir * hover;

        return ok_position(final_pos);
    }

    /* Fallback: re-raycast from current position if movement ray missed. */
    let to_center2 = Point3::origin() - rover_pos;
    let ray_dir2 = to_center2.normalize();
    let ray2 = Ray::new(rover_pos - ray_dir2 * 15.0, ray_dir2);

    if let Some(t) = mesh.cast_ray(&Isometry3::identity(), &ray2, 30.0, true) {
        let hit = ray2.origin + ray2.dir * t;
        let final_pos = hit - ray_dir2 * 0.08;
        return ok_position(final_pos);
    }

    ok_position(rover_pos)
}

fn ok_position(p: Point3<f32>) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&RoverTransform {
        position: [p.x, p.y, p.z],
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Raycasts from outside toward center; returns surface position and normal for waypoint placement.
#[wasm_bindgen]
pub fn get_surface_point_in_direction(
    dir_x: f32,
    dir_y: f32,
    dir_z: f32,
) -> Result<JsValue, JsValue> {
    let mesh_guard = ASTEROID_MESH.lock().unwrap();
    let mesh = mesh_guard
        .as_ref()
        .ok_or(JsValue::from_str("Collision mesh not loaded yet"))?;

    let dir = Vector3::new(dir_x, dir_y, dir_z);
    let dir = if dir.norm() > 0.001 {
        dir.normalize()
    } else {
        return Err(JsValue::from_str("Direction too small"));
    };

    let ray_origin = Point3::origin() + dir * 15.0;
    let ray_dir = -dir;
    let ray = Ray::new(ray_origin, ray_dir);

    if let Some(intersection) =
        mesh.cast_ray_and_get_normal(&Isometry3::identity(), &ray, 30.0, true)
    {
        let hit = ray.origin + ray.dir * intersection.toi;
        let hover = 0.02;
        let final_pos = hit - ray_dir * hover;
        let n = intersection.normal;
        serde_wasm_bindgen::to_value(&WaypointPlacement {
            position: [final_pos.x, final_pos.y, final_pos.z],
            normal: [n.x, n.y, n.z],
        })
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    } else {
        Err(JsValue::from_str("No surface hit in direction"))
    }
}
