//! Resumable training state: every trained tensor, the optimizer moments, and
//! where in the data the run had got to. One safetensors file, written atomically.
//!
//! Tensors are stored as `p.<name>`, `m.<name>`, `v.<name>`.

use std::collections::HashMap;
use std::path::Path;

use cf_io::{read_safetensors, write_safetensors};
use cf_model::{visit_params, Grads, Model};

use crate::optim::AdamW;

pub fn save_state(
    path: &Path,
    model: &mut Model,
    grads: &Grads,
    opt: &AdamW,
    step: usize,
    consumed: u64,
) -> Result<(), String> {
    let mut owned: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    visit_params(model, Some(grads), &mut |name, shape, p, g| {
        if g.is_empty() {
            return;
        }
        owned.push((format!("p.{name}"), shape.to_vec(), p.to_vec()));
        let zeros;
        let (m, v) = match opt.state.get(name) {
            Some((m, v)) => (m, v),
            None => {
                zeros = vec![0.0f32; p.len()];
                (&zeros, &zeros)
            }
        };
        owned.push((format!("m.{name}"), shape.to_vec(), m.clone()));
        owned.push((format!("v.{name}"), shape.to_vec(), v.clone()));
    });
    let refs: Vec<(String, Vec<usize>, &[f32])> = owned.iter().map(|(n, s, d)| (n.clone(), s.clone(), d.as_slice())).collect();

    let mut meta = HashMap::new();
    meta.insert("step".to_string(), step.to_string());
    meta.insert("consumed".to_string(), consumed.to_string());
    meta.insert("opt_step".to_string(), opt.step.to_string());
    if let Some(lc) = &model.lora_cfg {
        meta.insert("lora_rank".to_string(), lc.rank.to_string());
        meta.insert("lora_alpha".to_string(), lc.alpha.to_string());
        meta.insert("lora_targets".to_string(), lc.targets.join(","));
    }
    write_safetensors(path, &refs, meta)
}

/// Restore a saved state into `model` and `opt`. Attaches LoRA adapters first if
/// the file has them and the model does not. Returns `(step, consumed)`.
pub fn load_state(path: &Path, model: &mut Model, opt: &mut AdamW) -> Result<(usize, u64), String> {
    let loaded = read_safetensors(path)?;
    let meta = &loaded.metadata;
    let num = |k: &str| -> Result<u64, String> {
        meta.get(k).ok_or(format!("state file lacks '{k}'"))?.parse::<u64>().map_err(|_| format!("bad '{k}' in state file"))
    };

    if let Some(rank) = meta.get("lora_rank") {
        let rank: usize = rank.parse().map_err(|_| "bad lora_rank")?;
        let alpha: f32 = meta.get("lora_alpha").ok_or("state file lacks lora_alpha")?.parse().map_err(|_| "bad lora_alpha")?;
        let targets_s = meta.get("lora_targets").cloned().unwrap_or_default();
        let targets: Vec<&str> = targets_s.split(',').filter(|s| !s.is_empty()).collect();
        match &model.lora_cfg {
            None => model.add_lora(rank, alpha, &targets, 0)?,
            Some(lc) if lc.rank == rank && lc.alpha == alpha && lc.targets.iter().map(String::as_str).eq(targets.iter().copied()) => {}
            Some(_) => return Err("model's LoRA setup differs from the one in the state file".into()),
        }
    }

    let mut err: Option<String> = None;
    let mut restored = 0usize;
    let mut new_state: HashMap<String, (Vec<f32>, Vec<f32>)> = HashMap::new();
    visit_params(model, None, &mut |name, shape, p, _| {
        if err.is_some() {
            return;
        }
        let Some(t) = loaded.tensors.get(&format!("p.{name}")) else { return };
        if t.shape() != shape {
            err = Some(format!("'{name}' has shape {:?} in the state file, expected {:?}", t.shape(), shape));
            return;
        }
        p.copy_from_slice(t.data());
        restored += 1;
        if let (Some(m), Some(v)) = (loaded.tensors.get(&format!("m.{name}")), loaded.tensors.get(&format!("v.{name}"))) {
            new_state.insert(name.to_string(), (m.data().to_vec(), v.data().to_vec()));
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    let in_file = loaded.tensors.keys().filter(|k| k.starts_with("p.")).count();
    if restored != in_file {
        return Err(format!("state file has {in_file} parameter tensors but only {restored} matched this model"));
    }
    opt.state = new_state;
    opt.step = num("opt_step")?;
    Ok((num("step")? as usize, num("consumed")?))
}
