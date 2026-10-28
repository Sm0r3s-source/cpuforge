//! Hugging Face `config.json` for Llama-style models (Llama, SmolLM, Mistral, ...).

use cf_core::ops::rope::Llama3Scaling;
use serde::{Deserialize, Serialize};

fn default_eps() -> f32 {
    1e-5
}
fn default_theta() -> f64 {
    10000.0
}
fn default_max_pos() -> usize {
    2048
}
fn default_act() -> String {
    "silu".to_string()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LlamaConfig {
    /// `llama`, `mistral` or `qwen3` (absent for from-scratch models). Anything else is refused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_type: Option<String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_key_value_heads: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_dim: Option<usize>,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_scaling: Option<serde_json::Value>,
}

impl LlamaConfig {
    /// Parse and validate. Rejects variants this engine would silently get wrong.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let cfg: LlamaConfig = serde_json::from_str(text).map_err(|e| format!("bad config.json: {e}"))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_act != "silu" {
            return Err(format!("unsupported hidden_act '{}' (only silu)", self.hidden_act));
        }
        if self.attention_bias || self.mlp_bias {
            return Err("models with attention/mlp biases are not supported yet".into());
        }
        if let Some(m) = &self.model_type {
            if !["llama", "mistral", "qwen3"].contains(&m.as_str()) {
                return Err(format!("model_type '{m}' is not supported (llama, mistral and qwen3 are)"));
            }
        }
        self.llama3_scaling()?;
        let kv = self.kv_heads();
        if self.num_attention_heads == 0 || kv == 0 || !self.num_attention_heads.is_multiple_of(kv) {
            return Err("num_attention_heads must be a positive multiple of num_key_value_heads".into());
        }
        if !self.head_dim().is_multiple_of(2) {
            return Err("head_dim must be even for RoPE".into());
        }
        Ok(())
    }

    /// Qwen3 normalises each query/key head with an RMSNorm before RoPE.
    pub fn qk_norm(&self) -> bool {
        self.model_type.as_deref() == Some("qwen3")
    }

    /// `rope_scaling` of type `llama3`; every other scaling kind is an error rather than silently wrong numbers.
    pub fn llama3_scaling(&self) -> Result<Option<Llama3Scaling>, String> {
        let Some(v) = self.rope_scaling.as_ref().filter(|v| !v.is_null()) else { return Ok(None) };
        let kind = v.get("rope_type").or_else(|| v.get("type")).and_then(|k| k.as_str()).unwrap_or("");
        if kind != "llama3" {
            return Err(format!("rope_scaling type '{kind}' is not supported (only llama3)"));
        }
        let num = |k: &str| v.get(k).and_then(|x| x.as_f64()).ok_or_else(|| format!("rope_scaling is missing '{k}'"));
        let s = Llama3Scaling {
            factor: num("factor")?,
            low_freq_factor: num("low_freq_factor")?,
            high_freq_factor: num("high_freq_factor")?,
            original_max_position: num("original_max_position_embeddings")?,
        };
        if s.factor <= 0.0 || s.high_freq_factor <= s.low_freq_factor {
            return Err("rope_scaling values are inconsistent".into());
        }
        Ok(Some(s))
    }

    pub fn kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Approximate parameter count (embeddings counted once if tied).
    pub fn num_params(&self) -> usize {
        let (d, f, hd) = (self.hidden_size, self.intermediate_size, self.head_dim());
        let (h, kv) = (self.num_attention_heads, self.kv_heads());
        let qk = if self.qk_norm() { 2 * hd } else { 0 };
        let per_layer = d * h * hd * 2 + d * kv * hd * 2 + 3 * d * f + 2 * d + qk;
        let embed = self.vocab_size * d;
        embed * if self.tie_word_embeddings { 1 } else { 2 } + per_layer * self.num_hidden_layers + d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMOL: &str = r#"{
        "architectures": ["LlamaForCausalLM"], "hidden_size": 576, "intermediate_size": 1536,
        "num_attention_heads": 9, "num_hidden_layers": 30, "num_key_value_heads": 3,
        "vocab_size": 49152, "rms_norm_eps": 1e-05, "rope_theta": 10000.0,
        "tie_word_embeddings": true, "hidden_act": "silu", "max_position_embeddings": 2048
    }"#;

    #[test]
    fn parses_smollm_style_config() {
        let c = LlamaConfig::from_json(SMOL).unwrap();
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.kv_heads(), 3);
        // ~135M parameters
        let p = c.num_params();
        assert!((130_000_000..140_000_000).contains(&p), "{p}");
    }

    #[test]
    fn parses_llama32_style_rope_scaling() {
        let c = LlamaConfig::from_json(
            r#"{"model_type":"llama","vocab_size":10,"hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,
                "rope_scaling":{"factor":32.0,"high_freq_factor":4.0,"low_freq_factor":1.0,"original_max_position_embeddings":8192,"rope_type":"llama3"}}"#,
        )
        .unwrap();
        let s = c.llama3_scaling().unwrap().unwrap();
        assert_eq!((s.factor, s.low_freq_factor, s.high_freq_factor, s.original_max_position), (32.0, 1.0, 4.0, 8192.0));
        assert!(LlamaConfig::from_json(r#"{"vocab_size":10,"hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,"rope_scaling":{"rope_type":"llama3","factor":8.0}}"#).is_err());
    }

    #[test]
    fn parses_qwen3_style_config() {
        let c = LlamaConfig::from_json(
            r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","attention_bias":false,"head_dim":128,"hidden_act":"silu",
                "hidden_size":1024,"intermediate_size":3072,"max_position_embeddings":40960,"num_attention_heads":16,"num_hidden_layers":28,
                "num_key_value_heads":8,"rms_norm_eps":1e-06,"rope_scaling":null,"rope_theta":1000000,"sliding_window":null,
                "tie_word_embeddings":true,"use_sliding_window":false,"vocab_size":151936}"#,
        )
        .unwrap();
        assert!(c.qk_norm());
        assert_eq!(c.head_dim(), 128); // not hidden/heads = 64
        let p = c.num_params();
        assert!((590_000_000..610_000_000).contains(&p), "Qwen3-0.6B should be about 0.6B parameters, got {p}");
    }

    #[test]
    fn defaults_apply() {
        let c = LlamaConfig::from_json(
            r#"{"vocab_size":10,"hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2}"#,
        )
        .unwrap();
        assert_eq!(c.kv_heads(), 2);
        assert_eq!(c.rope_theta, 10000.0);
        assert!(!c.tie_word_embeddings);
    }

    #[test]
    fn rejects_unsupported_variants() {
        let base = |extra: &str| {
            format!(
                r#"{{"vocab_size":10,"hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,{extra}}}"#
            )
        };
        assert!(LlamaConfig::from_json(&base(r#""attention_bias": true"#)).is_err());
        assert!(LlamaConfig::from_json(&base(r#""hidden_act": "gelu""#)).is_err());
        assert!(LlamaConfig::from_json(&base(r#""rope_scaling": {"type": "linear", "factor": 2.0}"#)).is_err());
        assert!(LlamaConfig::from_json(&base(r#""rope_scaling": {"rope_type": "yarn", "factor": 2.0}"#)).is_err());
        assert!(LlamaConfig::from_json(&base(r#""model_type": "gemma3""#)).is_err());
        assert!(LlamaConfig::from_json(&base(r#""model_type": "qwen3_moe""#)).is_err());
        assert!(LlamaConfig::from_json(&base(r#""num_key_value_heads": 3"#)).is_err());
    }
}
