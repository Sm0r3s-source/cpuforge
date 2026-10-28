//! Tokenizers and datasets.
//!
//! Sequences are processed one at a time, so examples keep their natural length
//! (no padding). An example's `targets[i]` is the label for the output at
//! position `i`, already shifted, with `IGNORE_INDEX` where there is no loss.

use std::path::Path;

use cf_core::ops::loss::IGNORE_INDEX;
use cf_core::rng::Rng;

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, String>;
    fn decode(&self, ids: &[u32]) -> Result<String, String>;
    fn vocab_size(&self) -> usize;
    /// End-of-sequence id, if the tokenizer has one.
    fn eos_id(&self) -> Option<u32>;
}

/// Raw bytes as tokens (vocab 256). Handy for from-scratch experiments.
pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        Ok(text.bytes().map(|b| b as u32).collect())
    }
    fn decode(&self, ids: &[u32]) -> Result<String, String> {
        let bytes: Vec<u8> = ids.iter().map(|&i| i as u8).collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
    fn vocab_size(&self) -> usize {
        256
    }
    fn eos_id(&self) -> Option<u32> {
        None
    }
}

/// A Hugging Face `tokenizer.json`.
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
    eos: Option<u32>,
}

impl HfTokenizer {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| format!("cannot load {}: {e}", path.display()))?;
        let eos = ["<|endoftext|>", "</s>", "<|im_end|>", "<eos>", "<|end_of_text|>"]
            .iter()
            .find_map(|t| inner.token_to_id(t));
        Ok(Self { inner, eos })
    }
}

impl Tokenizer for HfTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let enc = self.inner.encode(text, false).map_err(|e| e.to_string())?;
        Ok(enc.get_ids().to_vec())
    }
    fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.inner.decode(ids, false).map_err(|e| e.to_string())
    }
    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
    fn eos_id(&self) -> Option<u32> {
        self.eos
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Example {
    pub tokens: Vec<u32>,
    pub targets: Vec<i64>,
}

#[derive(Clone, Debug, Default)]
pub struct Dataset {
    pub examples: Vec<Example>,
}

/// Turn a token sequence into an example predicting every next token.
fn lm_example(ids: &[u32]) -> Option<Example> {
    if ids.len() < 2 {
        return None;
    }
    Some(Example {
        tokens: ids[..ids.len() - 1].to_vec(),
        targets: ids[1..].iter().map(|&t| t as i64).collect(),
    })
}

impl Dataset {
    /// Cut one long token stream into windows of up to `seq_len` input tokens.
    pub fn from_stream(ids: &[u32], seq_len: usize) -> Dataset {
        let mut examples = Vec::new();
        let mut start = 0;
        while start + 1 < ids.len() {
            let end = (start + seq_len + 1).min(ids.len());
            if let Some(e) = lm_example(&ids[start..end]) {
                examples.push(e);
            }
            start += seq_len;
        }
        Dataset { examples }
    }

    /// JSON lines. Each line is either `{"text": "..."}` (loss on every token) or
    /// `{"prompt": "...", "completion": "..."}` (loss on the completion only).
    /// An end-of-sequence token is appended when the tokenizer has one.
    pub fn from_jsonl(text: &str, tok: &dyn Tokenizer, seq_len: usize) -> Result<Dataset, String> {
        let mut examples = Vec::new();
        for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
            let v: serde_json::Value = serde_json::from_str(line).map_err(|e| format!("line {}: {e}", n + 1))?;
            let field = |k: &str| v.get(k).and_then(|x| x.as_str());
            let (mut ids, prompt_len) = match (field("prompt"), field("completion"), field("text")) {
                (Some(p), Some(c), _) => {
                    let mut ids = tok.encode(p)?;
                    let plen = ids.len();
                    ids.extend(tok.encode(c)?);
                    (ids, plen)
                }
                (_, _, Some(t)) => (tok.encode(t)?, 0),
                _ => return Err(format!("line {}: expected \"text\" or \"prompt\"+\"completion\"", n + 1)),
            };
            if let Some(eos) = tok.eos_id() {
                ids.push(eos);
            }
            ids.truncate(seq_len + 1);
            if let Some(mut e) = lm_example(&ids) {
                // targets[i] predicts ids[i+1]; mask those that land inside the prompt.
                for (i, t) in e.targets.iter_mut().enumerate() {
                    if i + 1 < prompt_len {
                        *t = IGNORE_INDEX;
                    }
                }
                if e.targets.iter().any(|&t| t != IGNORE_INDEX) {
                    examples.push(e);
                }
            }
        }
        Ok(Dataset { examples })
    }

    /// Already-tokenized JSON lines, as produced by Brewery: `{"input_ids": [...], "labels": [...]}`
    /// with `labels` aligned to `input_ids` (Hugging Face convention, -100 = no loss). The shift
    /// by one happens here. Sequences longer than `seq_len + 1` tokens are cut.
    pub fn from_tokenized_jsonl(text: &str, seq_len: usize) -> Result<Dataset, String> {
        let mut examples = Vec::new();
        for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
            let v: serde_json::Value = serde_json::from_str(line).map_err(|e| format!("line {}: {e}", n + 1))?;
            let ints = |k: &str| -> Result<Vec<i64>, String> {
                v.get(k)
                    .and_then(|x| x.as_array())
                    .ok_or_else(|| format!("line {}: missing \"{k}\" array", n + 1))?
                    .iter()
                    .map(|x| x.as_i64().ok_or_else(|| format!("line {}: \"{k}\" must hold integers", n + 1)))
                    .collect()
            };
            let (ids, labels) = (ints("input_ids")?, ints("labels")?);
            if ids.len() != labels.len() {
                return Err(format!("line {}: input_ids has {} entries but labels has {}", n + 1, ids.len(), labels.len()));
            }
            if let Some(bad) = ids.iter().find(|&&i| !(0..=u32::MAX as i64).contains(&i)) {
                return Err(format!("line {}: invalid token id {bad}", n + 1));
            }
            let keep = ids.len().min(seq_len + 1);
            if keep < 2 {
                continue;
            }
            let e = Example {
                tokens: ids[..keep - 1].iter().map(|&i| i as u32).collect(),
                targets: labels[1..keep].iter().map(|&l| if l < 0 { IGNORE_INDEX } else { l }).collect(),
            };
            if e.targets.iter().any(|&t| t != IGNORE_INDEX) {
                examples.push(e);
            }
        }
        Ok(Dataset { examples })
    }

    /// Largest token id used as an input or a target (to check against the model's vocabulary).
    pub fn max_token_id(&self) -> Option<i64> {
        self.examples.iter().flat_map(|e| e.tokens.iter().map(|&t| t as i64).chain(e.targets.iter().copied())).max()
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Total input tokens across all examples.
    pub fn total_tokens(&self) -> usize {
        self.examples.iter().map(|e| e.tokens.len()).sum()
    }

    /// Move about `fraction` of the examples into a new validation set, picking
    /// evenly spaced ones so it covers the whole corpus (taking the tail would
    /// validate on whichever file happens to come last). Deterministic.
    pub fn split_off_val(&mut self, fraction: f32) -> Dataset {
        if fraction <= 0.0 {
            return Dataset::default();
        }
        let stride = ((1.0 / fraction).round() as usize).max(2);
        let (mut train, mut val) = (Vec::new(), Vec::new());
        for (i, e) in std::mem::take(&mut self.examples).into_iter().enumerate() {
            if i % stride == stride - 1 {
                val.push(e);
            } else {
                train.push(e);
            }
        }
        self.examples = train;
        Dataset { examples: val }
    }

    /// Shuffled visiting order for an epoch; deterministic in (seed, epoch), so a
    /// resumed run only needs to know how many examples it has consumed.
    pub fn order(&self, seed: u64, epoch: u64) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.examples.len()).collect();
        let mut rng = Rng::new(seed ^ epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        for i in (1..idx.len()).rev() {
            idx.swap(i, rng.below(i + 1));
        }
        idx
    }
}

#[cfg(test)]
#[allow(clippy::err_expect)]
mod tests {
    use super::*;

    #[test]
    fn byte_tokenizer_roundtrip() {
        let t = ByteTokenizer;
        let ids = t.encode("héllo").unwrap();
        assert_eq!(t.decode(&ids).unwrap(), "héllo");
        assert_eq!(t.vocab_size(), 256);
    }

    #[test]
    fn stream_windows_are_shifted_and_cover_everything() {
        let ids: Vec<u32> = (0..23).collect();
        let d = Dataset::from_stream(&ids, 8);
        assert_eq!(d.examples[0].tokens, (0..8).collect::<Vec<u32>>());
        assert_eq!(d.examples[0].targets, (1..9).map(|x| x as i64).collect::<Vec<_>>());
        // Windows overlap by one token so every next-token pair is used exactly once.
        let pairs: usize = d.examples.iter().map(|e| e.tokens.len()).sum();
        assert_eq!(pairs, 22);
        assert!(d.examples.iter().all(|e| e.tokens.len() <= 8 && e.tokens.len() == e.targets.len()));
    }

    #[test]
    fn tokenized_jsonl_shifts_labels_and_keeps_masks() {
        let text = r#"{"input_ids":[5,6,7,8],"labels":[-100,-100,7,8]}
{"input_ids":[1],"labels":[1]}
{"input_ids":[9,9,9],"labels":[-100,-100,-100]}
{"input_ids":[1,2,3,4,5,6],"labels":[1,2,3,4,5,6]}
"#;
        let d = Dataset::from_tokenized_jsonl(text, 3).unwrap();
        assert_eq!(d.len(), 2); // too short and fully masked lines are dropped
        // position i predicts label[i+1]; the first label is never a target
        assert_eq!(d.examples[0].tokens, vec![5, 6, 7]);
        assert_eq!(d.examples[0].targets, vec![IGNORE_INDEX, 7, 8]);
        // seq_len 3 means at most 3 input tokens (4 ids)
        assert_eq!(d.examples[1].tokens, vec![1, 2, 3]);
        assert_eq!(d.examples[1].targets, vec![2, 3, 4]);
        assert_eq!(d.max_token_id(), Some(8));
    }

    #[test]
    fn tokenized_jsonl_errors_are_specific() {
        let e = |s: &str| Dataset::from_tokenized_jsonl(s, 8).err().expect("should fail");
        assert!(e(r#"{"input_ids":[1,2],"labels":[1]}"#).contains("labels has 1"));
        assert!(e(r#"{"input_ids":[1,2]}"#).contains("missing \"labels\""));
        assert!(e(r#"{"input_ids":[1,-2],"labels":[1,2]}"#).contains("invalid token id"));
        assert!(e(r#"{"input_ids":[1,"x"],"labels":[1,2]}"#).contains("integers"));
        assert!(e("nope").contains("line 1"));
    }

    #[test]
    fn jsonl_masks_the_prompt() {
        let text = r#"{"prompt": "ab", "completion": "cde"}
{"text": "xyz"}

"#;
        let d = Dataset::from_jsonl(text, &ByteTokenizer, 64).unwrap();
        assert_eq!(d.len(), 2);
        let e = &d.examples[0]; // ids: a b c d e
        assert_eq!(e.tokens, vec![97, 98, 99, 100]);
        assert_eq!(e.targets, vec![IGNORE_INDEX, 99, 100, 101]);
        // Position 0 predicts 'b', which is still prompt; position 1 predicts 'c', the first completion token.
        let t = &d.examples[1];
        assert_eq!(t.targets, vec![121, 122]);
    }

    #[test]
    fn jsonl_truncates_and_reports_bad_lines() {
        let long = format!(r#"{{"text": "{}"}}"#, "a".repeat(100));
        let d = Dataset::from_jsonl(&long, &ByteTokenizer, 10).unwrap();
        assert_eq!(d.examples[0].tokens.len(), 10);
        assert!(Dataset::from_jsonl("not json", &ByteTokenizer, 10).err().expect("bad json").contains("line 1"));
        assert!(Dataset::from_jsonl(r#"{"foo": 1}"#, &ByteTokenizer, 10).is_err());
        // A prompt that fills the whole window leaves nothing to learn: skipped.
        let only_prompt = format!(r#"{{"prompt": "{}", "completion": "z"}}"#, "p".repeat(50));
        assert!(Dataset::from_jsonl(&only_prompt, &ByteTokenizer, 10).unwrap().is_empty());
    }

    #[test]
    fn order_is_a_deterministic_permutation_that_changes_per_epoch() {
        let d = Dataset::from_stream(&(0..400).collect::<Vec<u32>>(), 8);
        let a = d.order(1, 0);
        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(sorted, (0..d.len()).collect::<Vec<_>>());
        assert_eq!(a, d.order(1, 0));
        assert_ne!(a, d.order(1, 1));
        assert_ne!(a, d.order(2, 0));
    }

    #[test]
    fn validation_split_keeps_at_least_one_training_example() {
        let mut d = Dataset::from_stream(&(0..100).collect::<Vec<u32>>(), 10);
        let n = d.len();
        let val = d.split_off_val(0.2);
        assert_eq!(d.len() + val.len(), n);
        assert!(!val.is_empty());
        let mut tiny = Dataset::from_stream(&[1, 2, 3], 8);
        let v = tiny.split_off_val(0.9);
        assert_eq!(tiny.len(), 1);
        assert_eq!(v.len(), 0);
        assert!(Dataset::from_stream(&[1, 2, 3], 8).split_off_val(0.0).is_empty());
    }

    #[test]
    fn validation_examples_are_spread_across_the_corpus() {
        let ids: Vec<u32> = (0..2000).collect();
        let mut d = Dataset::from_stream(&ids, 10);
        let val = d.split_off_val(0.1);
        let firsts: Vec<u32> = val.examples.iter().map(|e| e.tokens[0]).collect();
        assert!(firsts.iter().any(|&t| t < 500), "no validation window from the first quarter");
        assert!(firsts.iter().any(|&t| t > 1500), "no validation window from the last quarter");
        assert!((15..=25).contains(&val.len()), "{}", val.len());
        // No example is in both sets.
        assert!(d.examples.iter().all(|e| !firsts.contains(&e.tokens[0])));
    }

    #[test]
    fn hf_tokenizer_loads_from_json() {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;
        let vocab = [("[UNK]", 0u32), ("hello", 1), ("world", 2), ("<|endoftext|>", 3)]
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect(); // the builder wants an ahash map; collect() infers it
        let model = WordLevel::builder().vocab(vocab).unk_token("[UNK]".into()).build().unwrap();
        let mut tok = tokenizers::Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(Whitespace {}));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        tok.save(&path, false).unwrap();

        let t = HfTokenizer::from_file(&path).unwrap();
        assert_eq!(t.encode("hello world").unwrap(), vec![1, 2]);
        assert_eq!(t.encode("hello nowhere").unwrap(), vec![1, 0]);
        assert_eq!(t.eos_id(), Some(3));
        assert_eq!(t.vocab_size(), 4);
        let d = Dataset::from_jsonl(r#"{"text": "hello world"}"#, &t, 16).unwrap();
        assert_eq!(d.examples[0].tokens, vec![1, 2]);
        assert_eq!(d.examples[0].targets, vec![2, 3]); // ends with eos
        assert!(HfTokenizer::from_file(&dir.path().join("missing.json")).is_err());
    }
}
