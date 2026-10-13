//! safetensors reading (mmap) and writing, as f32 tensors.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use cf_core::Tensor;

use crate::dtype::{bf16_to_f32, f16_to_f32};

pub struct Loaded {
    pub tensors: HashMap<String, Tensor>,
    pub metadata: HashMap<String, String>,
}

/// Read every tensor in a `.safetensors` file, converting f32/bf16/f16 to f32.
pub fn read_safetensors(path: &Path) -> Result<Loaded, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    // SAFETY: the mapping is read-only and we copy everything out before returning. As
    // with any mmap, modifying the file while it is mapped is undefined, which is the
    // accepted trade-off for not reading multi-GB checkpoints through a buffer.
    let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("cannot mmap {}: {e}", path.display()))?;
    let st = SafeTensors::deserialize(&map).map_err(|e| format!("{}: {e}", path.display()))?;
    let metadata = SafeTensors::read_metadata(&map)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .1
        .metadata()
        .clone()
        .unwrap_or_default();

    let mut tensors = HashMap::new();
    for (name, view) in st.tensors() {
        let bytes = view.data();
        let data: Vec<f32> = match view.dtype() {
            Dtype::F32 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            Dtype::BF16 => bytes.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            Dtype::F16 => bytes.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            other => return Err(format!("tensor '{name}' has unsupported dtype {other:?}")),
        };
        tensors.insert(name, Tensor::from_vec(data, view.shape()));
    }
    Ok(Loaded { tensors, metadata })
}

/// Write f32 tensors atomically (temp file, then rename).
pub fn write_safetensors(
    path: &Path,
    tensors: &[(String, Vec<usize>, &[f32])],
    metadata: HashMap<String, String>,
) -> Result<(), String> {
    let buffers: Vec<Vec<u8>> = tensors.iter().map(|(_, _, d)| d.iter().flat_map(|v| v.to_le_bytes()).collect()).collect();
    let mut views = Vec::with_capacity(tensors.len());
    for ((name, shape, _), bytes) in tensors.iter().zip(&buffers) {
        let view = TensorView::new(Dtype::F32, shape.clone(), bytes).map_err(|e| format!("tensor '{name}': {e}"))?;
        views.push((name.clone(), view));
    }
    let tmp = path.with_extension("tmp");
    let meta = if metadata.is_empty() { None } else { Some(metadata) };
    safetensors::serialize_to_file(views, meta, &tmp).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("cannot move {} into place: {e}", tmp.display()))
}

#[cfg(test)]
#[allow(clippy::err_expect)]
mod tests {
    use super::*;

    #[test]
    fn f32_roundtrip_with_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.safetensors");
        let a = [1.0f32, -2.5, 3.25, 0.0, 7.0, 8.0];
        let b = [0.5f32; 4];
        let mut meta = HashMap::new();
        meta.insert("step".to_string(), "42".to_string());
        write_safetensors(&p, &[("a".into(), vec![2, 3], &a), ("b".into(), vec![4], &b)], meta).unwrap();
        let l = read_safetensors(&p).unwrap();
        assert_eq!(l.tensors["a"].shape(), &[2, 3]);
        assert_eq!(l.tensors["a"].data(), &a);
        assert_eq!(l.tensors["b"].data(), &b);
        assert_eq!(l.metadata["step"], "42");
        assert!(!dir.path().join("a.tmp").exists());
    }

    #[test]
    fn reads_bf16_and_f16_files() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("h.safetensors");
        // bf16: 1.0, -2.0, 3.140625   f16: 1.0, -2.0, 65504
        let bf: Vec<u8> = [0x3F80u16, 0xC000, 0x4049].iter().flat_map(|v| v.to_le_bytes()).collect();
        let hf: Vec<u8> = [0x3C00u16, 0xC000, 0x7BFF].iter().flat_map(|v| v.to_le_bytes()).collect();
        let views = vec![
            ("bf".to_string(), TensorView::new(Dtype::BF16, vec![3], &bf).unwrap()),
            ("hf".to_string(), TensorView::new(Dtype::F16, vec![3], &hf).unwrap()),
        ];
        safetensors::serialize_to_file(views, None, &p).unwrap();
        let l = read_safetensors(&p).unwrap();
        assert_eq!(l.tensors["bf"].data(), &[1.0, -2.0, 3.140625]);
        assert_eq!(l.tensors["hf"].data(), &[1.0, -2.0, 65504.0]);
    }

    #[test]
    fn rejects_integer_tensors_and_missing_files() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("i.safetensors");
        let bytes = [0u8; 8];
        let views = vec![("ids".to_string(), TensorView::new(Dtype::I64, vec![1], &bytes).unwrap())];
        safetensors::serialize_to_file(views, None, &p).unwrap();
        assert!(read_safetensors(&p).err().expect("should fail").contains("unsupported dtype"));
        assert!(read_safetensors(&dir.path().join("nope")).is_err());
    }
}
