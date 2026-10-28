//! Hugging Face model directories and LoRA adapter files.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use cf_core::Tensor;
use cf_model::{visit_params, LlamaConfig, Model};

use crate::st::{read_safetensors, write_safetensors};

/// Load `config.json` plus `model.safetensors` (or a sharded
/// `model.safetensors.index.json`) from a directory.
pub fn load_model(dir: &Path, max_seq: usize) -> Result<Model, String> {
    if max_seq == 0 {
        return Err("max_seq must be positive".into());
    }
    let cfg_path = dir.join("config.json");
    let text = std::fs::read_to_string(&cfg_path).map_err(|e| format!("cannot read {}: {e}", cfg_path.display()))?;
    let cfg = LlamaConfig::from_json(&text)?;

    let single = dir.join("model.safetensors");
    let index = dir.join("model.safetensors.index.json");
    let files: Vec<_> = if single.exists() {
        vec![single]
    } else if index.exists() {
        let text = std::fs::read_to_string(&index).map_err(|e| format!("cannot read {}: {e}", index.display()))?;
        let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("bad index json: {e}"))?;
        let map = json["weight_map"].as_object().ok_or("index json has no weight_map")?;
        let shards: BTreeSet<&str> = map.values().filter_map(|v| v.as_str()).collect();
        shards.into_iter().map(|s| dir.join(s)).collect()
    } else {
        return Err(format!("no model.safetensors or model.safetensors.index.json in {}", dir.display()));
    };

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for f in files {
        tensors.extend(read_safetensors(&f)?.tensors);
    }
    Model::from_named(cfg, max_seq, tensors)
}

/// Write a Hugging Face-style directory (`config.json` + `model.safetensors`).
/// LoRA adapters are not included; call `merge_lora` first to bake them in.
pub fn save_model(dir: &Path, model: &mut Model) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let cfg = serde_json::to_string_pretty(&model.cfg).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("config.json"), cfg).map_err(|e| e.to_string())?;

    let mut owned: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    visit_params(model, None, &mut |name, shape, data, _| {
        if !name.contains(".lora_") {
            owned.push((name.to_string(), shape.to_vec(), data.to_vec()));
        }
    });
    let refs: Vec<(String, Vec<usize>, &[f32])> = owned.iter().map(|(n, s, d)| (n.clone(), s.clone(), d.as_slice())).collect();
    write_safetensors(&dir.join("model.safetensors"), &refs, HashMap::new())
}

/// Save only the LoRA tensors, plus what is needed to rebuild the adapter structure.
pub fn save_adapter(path: &Path, model: &mut Model) -> Result<(), String> {
    let lc = model.lora_cfg.clone().ok_or("model has no LoRA adapters")?;
    let mut owned: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    visit_params(model, None, &mut |name, shape, data, _| {
        if name.contains(".lora_") {
            owned.push((name.to_string(), shape.to_vec(), data.to_vec()));
        }
    });
    let refs: Vec<(String, Vec<usize>, &[f32])> = owned.iter().map(|(n, s, d)| (n.clone(), s.clone(), d.as_slice())).collect();
    let mut meta = HashMap::new();
    meta.insert("lora_rank".to_string(), lc.rank.to_string());
    meta.insert("lora_alpha".to_string(), lc.alpha.to_string());
    meta.insert("lora_targets".to_string(), lc.targets.join(","));
    write_safetensors(path, &refs, meta)
}

/// Short LoRA target name (`q`) <-> Hugging Face module name (`q_proj`).
fn proj_name(short: &str) -> String {
    format!("{short}_proj")
}

/// Write the adapter in the layout Hugging Face PEFT reads (`adapter_config.json` +
/// `adapter_model.safetensors`), so other tools can load, merge or publish it.
pub fn save_peft_adapter(dir: &Path, model: &mut Model, base_model: &str) -> Result<(), String> {
    let lc = model.lora_cfg.clone().ok_or("model has no LoRA adapters")?;
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let mut owned: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    visit_params(model, None, &mut |name, shape, data, _| {
        if name.contains(".lora_") {
            owned.push((format!("base_model.model.{name}.weight"), shape.to_vec(), data.to_vec()));
        }
    });
    let refs: Vec<(String, Vec<usize>, &[f32])> = owned.iter().map(|(n, s, d)| (n.clone(), s.clone(), d.as_slice())).collect();
    let mut meta = HashMap::new();
    meta.insert("format".to_string(), "pt".to_string());
    write_safetensors(&dir.join("adapter_model.safetensors"), &refs, meta)?;
    let cfg = serde_json::json!({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base_model,
        "r": lc.rank, "lora_alpha": lc.alpha, "lora_dropout": 0.0, "bias": "none",
        "target_modules": lc.targets.iter().map(|s| proj_name(s)).collect::<Vec<_>>(),
        "fan_in_fan_out": false, "inference_mode": true, "use_rslora": false, "use_dora": false,
    });
    std::fs::write(dir.join("adapter_config.json"), serde_json::to_string_pretty(&cfg).unwrap()).map_err(|e| e.to_string())
}

/// Attach adapters to `model` from a PEFT-style directory (the inverse of `save_peft_adapter`).
pub fn load_peft_adapter(model: &mut Model, dir: &Path) -> Result<(), String> {
    let cfg_text = std::fs::read_to_string(dir.join("adapter_config.json")).map_err(|e| format!("cannot read adapter_config.json in {}: {e}", dir.display()))?;
    let cfg: serde_json::Value = serde_json::from_str(&cfg_text).map_err(|e| format!("bad adapter_config.json: {e}"))?;
    if cfg["peft_type"] != "LORA" {
        return Err("only LoRA adapters are supported".into());
    }
    if cfg["use_dora"] == true || cfg["use_rslora"] == true {
        return Err("DoRA / rsLoRA adapters are not supported".into());
    }
    let rank = cfg["r"].as_u64().ok_or("adapter_config.json lacks r")? as usize;
    let alpha = cfg["lora_alpha"].as_f64().ok_or("adapter_config.json lacks lora_alpha")? as f32;
    let mods: Vec<String> = cfg["target_modules"].as_array().ok_or("adapter_config.json lacks target_modules")?.iter().filter_map(|v| v.as_str().map(String::from)).collect();
    let targets: Vec<&str> = mods.iter().map(|m| m.strip_suffix("_proj").unwrap_or(m)).collect();
    model.add_lora(rank, alpha, &targets, 0)?;
    let loaded = read_safetensors(&dir.join("adapter_model.safetensors"))?;
    let (mut err, mut seen) = (None::<String>, 0usize);
    visit_params(model, None, &mut |name, shape, data, _| {
        if !name.contains(".lora_") || err.is_some() {
            return;
        }
        let key = format!("base_model.model.{name}.weight");
        match loaded.tensors.get(&key) {
            None => err = Some(format!("adapter file is missing '{key}'")),
            Some(t) if t.shape() != shape => err = Some(format!("adapter tensor '{key}' has shape {:?}, expected {:?}", t.shape(), shape)),
            Some(t) => {
                data.copy_from_slice(t.data());
                seen += 1;
            }
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    if seen != loaded.tensors.len() {
        return Err(format!("adapter file has {} tensors but the model uses {seen}", loaded.tensors.len()));
    }
    Ok(())
}

/// Attach adapters to `model` and fill them from a file written by `save_adapter`.
pub fn load_adapter(model: &mut Model, path: &Path) -> Result<(), String> {
    let loaded = read_safetensors(path)?;
    let get = |k: &str| loaded.metadata.get(k).cloned().ok_or(format!("adapter file lacks metadata '{k}'"));
    let rank: usize = get("lora_rank")?.parse().map_err(|_| "bad lora_rank")?;
    let alpha: f32 = get("lora_alpha")?.parse().map_err(|_| "bad lora_alpha")?;
    let targets_s = get("lora_targets")?;
    let targets: Vec<&str> = targets_s.split(',').filter(|s| !s.is_empty()).collect();
    model.add_lora(rank, alpha, &targets, 0)?;

    let mut err: Option<String> = None;
    let mut seen = 0usize;
    visit_params(model, None, &mut |name, shape, data, _| {
        if !name.contains(".lora_") || err.is_some() {
            return;
        }
        match loaded.tensors.get(name) {
            None => err = Some(format!("adapter file is missing '{name}'")),
            Some(t) if t.shape() != shape => {
                err = Some(format!("adapter tensor '{name}' has shape {:?}, expected {:?}", t.shape(), shape))
            }
            Some(t) => {
                data.copy_from_slice(t.data());
                seen += 1;
            }
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    if seen != loaded.tensors.len() {
        return Err(format!("adapter file has {} tensors but the model uses {seen}", loaded.tensors.len()));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::err_expect)]
mod tests {
    use super::*;
    use cf_core::rng::Rng;

    fn cfg() -> LlamaConfig {
        LlamaConfig::from_json(
            r#"{"vocab_size":13,"hidden_size":16,"intermediate_size":24,"num_hidden_layers":2,
                "num_attention_heads":4,"num_key_value_heads":2,"max_position_embeddings":32,
                "tie_word_embeddings":true}"#,
        )
        .unwrap()
    }

    const TOK: [u32; 6] = [3, 1, 4, 1, 5, 9];

    #[test]
    fn save_then_load_gives_identical_logits() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = Model::random(cfg(), 32, 1, 0.3);
        save_model(dir.path(), &mut m).unwrap();
        let loaded = load_model(dir.path(), 16).unwrap();
        assert_eq!(m.logits(&TOK), loaded.logits(&TOK));
    }

    #[test]
    fn sharded_checkpoints_load() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = Model::random(cfg(), 32, 2, 0.3);
        save_model(dir.path(), &mut m).unwrap();
        // Split the single file into two shards by tensor name.
        let all = read_safetensors(&dir.path().join("model.safetensors")).unwrap().tensors;
        std::fs::remove_file(dir.path().join("model.safetensors")).unwrap();
        let mut names: Vec<_> = all.keys().cloned().collect();
        names.sort();
        let (a, b) = names.split_at(names.len() / 2);
        let mut weight_map = serde_json::Map::new();
        for (shard, group) in [("s1.safetensors", a), ("s2.safetensors", b)] {
            let refs: Vec<_> = group.iter().map(|n| (n.clone(), all[n].shape().to_vec(), all[n].data())).collect();
            write_safetensors(&dir.path().join(shard), &refs, HashMap::new()).unwrap();
            for n in group {
                weight_map.insert(n.clone(), serde_json::Value::String(shard.into()));
            }
        }
        let index = serde_json::json!({ "metadata": {}, "weight_map": weight_map });
        std::fs::write(dir.path().join("model.safetensors.index.json"), index.to_string()).unwrap();
        let loaded = load_model(dir.path(), 16).unwrap();
        assert_eq!(m.logits(&TOK), loaded.logits(&TOK));
    }

    #[test]
    fn tied_checkpoint_with_extra_lm_head_still_loads() {
        // Many tied checkpoints also ship lm_head.weight; it must be ignored, not fail.
        let dir = tempfile::tempdir().unwrap();
        let mut m = Model::random(cfg(), 32, 3, 0.3);
        save_model(dir.path(), &mut m).unwrap();
        let mut all = read_safetensors(&dir.path().join("model.safetensors")).unwrap().tensors;
        let embed = all["model.embed_tokens.weight"].clone();
        all.insert("lm_head.weight".into(), embed);
        let refs: Vec<_> = all.iter().map(|(n, t)| (n.clone(), t.shape().to_vec(), t.data())).collect();
        write_safetensors(&dir.path().join("model.safetensors"), &refs, HashMap::new()).unwrap();
        assert!(load_model(dir.path(), 16).is_ok());
    }

    #[test]
    fn missing_files_give_clear_errors() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_model(dir.path(), 8).err().expect("no config").contains("config.json"));
        std::fs::write(dir.path().join("config.json"), serde_json::to_string(&cfg()).unwrap()).unwrap();
        assert!(load_model(dir.path(), 8).err().expect("no weights").contains("model.safetensors"));
        assert!(load_model(dir.path(), 0).is_err());
    }

    #[test]
    fn adapter_roundtrip_restores_behaviour() {
        let dir = tempfile::tempdir().unwrap();
        let mut base = Model::random(cfg(), 32, 4, 0.3);
        base.add_lora(3, 6.0, &["q", "v", "down"], 7).unwrap();
        let mut rng = Rng::new(8);
        for l in &mut base.layers {
            for lin in [&mut l.wq, &mut l.wv, &mut l.down] {
                lin.lora.as_mut().unwrap().b.data_mut().iter_mut().for_each(|v| *v = rng.normal() * 0.3);
            }
        }
        let want = base.logits(&TOK);
        let path = dir.path().join("adapter.safetensors");
        save_adapter(&path, &mut base).unwrap();

        // Fresh copy of the same base weights, adapter restored from disk.
        let mut fresh = Model::random(cfg(), 32, 4, 0.3);
        assert_ne!(fresh.logits(&TOK), want);
        load_adapter(&mut fresh, &path).unwrap();
        assert_eq!(fresh.logits(&TOK), want);
        assert_eq!(fresh.lora_cfg, base.lora_cfg);
    }

    #[test]
    fn peft_adapter_roundtrips_with_peft_names_and_config() {
        let dir = tempfile::tempdir().unwrap();
        let mut base = Model::random(cfg(), 32, 4, 0.3);
        base.add_lora(3, 6.0, &["q", "v", "down"], 7).unwrap();
        let mut rng = Rng::new(8);
        for l in &mut base.layers {
            for lin in [&mut l.wq, &mut l.wv, &mut l.down] {
                lin.lora.as_mut().unwrap().b.data_mut().iter_mut().for_each(|v| *v = rng.normal() * 0.3);
            }
        }
        let want = base.logits(&TOK);
        save_peft_adapter(dir.path(), &mut base, "org/base").unwrap();

        let c: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.path().join("adapter_config.json")).unwrap()).unwrap();
        assert_eq!((c["peft_type"].as_str(), c["r"].as_u64(), c["lora_alpha"].as_f64()), (Some("LORA"), Some(3), Some(6.0)));
        assert_eq!(c["base_model_name_or_path"], "org/base");
        assert_eq!(c["target_modules"], serde_json::json!(["q_proj", "v_proj", "down_proj"]));
        let names: Vec<String> = read_safetensors(&dir.path().join("adapter_model.safetensors")).unwrap().tensors.keys().cloned().collect();
        assert!(names.contains(&"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string()), "{names:?}");
        assert!(names.contains(&"base_model.model.model.layers.1.mlp.down_proj.lora_B.weight".to_string()));
        assert_eq!(names.len(), 2 * 3 * 2); // A and B, three modules, two layers

        let mut fresh = Model::random(cfg(), 32, 4, 0.3);
        load_peft_adapter(&mut fresh, dir.path()).unwrap();
        assert_eq!(fresh.logits(&TOK), want);
        std::fs::write(dir.path().join("adapter_config.json"), c.to_string().replace("LORA", "IA3")).unwrap();
        let mut again = Model::random(cfg(), 32, 4, 0.3);
        assert!(load_peft_adapter(&mut again, dir.path()).is_err());
    }

    #[test]
    fn qwen3_model_saves_and_loads_with_norms_and_model_type() {
        let dir = tempfile::tempdir().unwrap();
        let qcfg = LlamaConfig::from_json(
            r#"{"model_type":"qwen3","vocab_size":13,"hidden_size":16,"intermediate_size":24,"num_hidden_layers":2,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":6,"max_position_embeddings":32,"tie_word_embeddings":true}"#,
        )
        .unwrap();
        let mut m = Model::random(qcfg, 32, 9, 0.3);
        m.layers[1].k_norm.as_mut().unwrap()[2] = 1.7;
        save_model(dir.path(), &mut m).unwrap();
        let text = std::fs::read_to_string(dir.path().join("config.json")).unwrap();
        assert!(text.contains(r#""model_type": "qwen3""#), "{text}");
        let loaded = load_model(dir.path(), 16).unwrap();
        assert_eq!(m.logits(&TOK), loaded.logits(&TOK));
        assert_eq!(loaded.layers[1].k_norm.as_ref().unwrap()[2], 1.7);
    }

    #[test]
    fn adapter_for_a_different_architecture_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let mut a = Model::random(cfg(), 32, 5, 0.3);
        a.add_lora(2, 4.0, &["q"], 1).unwrap();
        let path = dir.path().join("a.safetensors");
        save_adapter(&path, &mut a).unwrap();

        let wide = LlamaConfig::from_json(
            r#"{"vocab_size":13,"hidden_size":32,"intermediate_size":24,"num_hidden_layers":2,
                "num_attention_heads":4,"num_key_value_heads":2,"max_position_embeddings":32}"#,
        )
        .unwrap();
        let mut b = Model::random(wide, 32, 6, 0.3);
        assert!(load_adapter(&mut b, &path).err().expect("shape mismatch").contains("shape"));
        let mut no_lora = Model::random(cfg(), 32, 7, 0.3);
        assert!(save_adapter(&path, &mut no_lora).is_err());
    }
}
