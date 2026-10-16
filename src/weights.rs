//! Memory-mapped `.safetensors` reader with on-the-fly BF16/F16 -> F32 conversion.

use anyhow::{Context, Result, anyhow, bail, ensure};
use memmap2::Mmap;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F16,
    F32,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    start: usize,
    end: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct Weights {
    mmap: Mmap,
    data_start: usize,
    index: HashMap<String, TensorInfo>,
}

#[derive(Deserialize)]
struct RawInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

impl Weights {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the file is opened read-only and not modified while mapped.
        let mmap = unsafe { Mmap::map(&file)? };
        ensure!(mmap.len() >= 8, "safetensors file too small");
        // Checked arithmetic throughout: a corrupt header must produce an error, not a wrapped
        // (release-mode) offset and a panic.
        let hdr_len = usize::try_from(u64::from_le_bytes(mmap[..8].try_into().unwrap()))
            .ok()
            .filter(|&n| n <= mmap.len() - 8)
            .ok_or_else(|| anyhow!("corrupt safetensors header length in {}", path.display()))?;
        let data_start = 8 + hdr_len;
        let header: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&mmap[8..data_start]).with_context(|| format!("parsing the header of {}", path.display()))?;
        let mut index = HashMap::with_capacity(header.len());
        for (name, val) in header {
            if name == "__metadata__" {
                continue;
            }
            let raw: RawInfo = serde_json::from_value(val).with_context(|| format!("tensor {name}"))?;
            let dtype = match raw.dtype.as_str() {
                "BF16" => Dtype::Bf16,
                "F16" => Dtype::F16,
                "F32" => Dtype::F32,
                other => bail!("tensor {name}: unsupported dtype {other}"),
            };
            let [start, end] = raw.data_offsets;
            let elem = match dtype {
                Dtype::F32 => 4,
                _ => 2,
            };
            ensure!(
                end >= start && data_start.checked_add(end).is_some_and(|e| e <= mmap.len()),
                "tensor {name}: bad offsets"
            );
            let nbytes = raw.shape.iter().try_fold(elem, |acc: usize, &d| acc.checked_mul(d));
            ensure!(nbytes == Some(end - start), "tensor {name}: byte size does not match shape");
            index.insert(name, TensorInfo { dtype, shape: raw.shape, start, end });
        }
        Ok(Self { mmap, data_start, index })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.index.get(name).ok_or_else(|| anyhow!("missing tensor `{name}`"))
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.index.keys()
    }

    fn bytes(&self, info: &TensorInfo) -> &[u8] {
        &self.mmap[self.data_start + info.start..self.data_start + info.end]
    }

    /// Whole tensor as `f32`, converting from the stored dtype.
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.info(name)?;
        let bytes = self.bytes(info);
        let n = info.numel();
        let mut out = vec![0f32; n];
        match info.dtype {
            Dtype::F32 => out
                .par_chunks_mut(1 << 16)
                .zip(bytes.par_chunks(4 << 16))
                .for_each(|(o, b)| {
                    for (d, s) in o.iter_mut().zip(b.chunks_exact(4)) {
                        *d = f32::from_le_bytes(s.try_into().unwrap());
                    }
                }),
            Dtype::Bf16 => out
                .par_chunks_mut(1 << 16)
                .zip(bytes.par_chunks(2 << 16))
                .for_each(|(o, b)| {
                    for (d, s) in o.iter_mut().zip(b.chunks_exact(2)) {
                        *d = bf16_to_f32(u16::from_le_bytes([s[0], s[1]]));
                    }
                }),
            Dtype::F16 => out
                .par_chunks_mut(1 << 16)
                .zip(bytes.par_chunks(2 << 16))
                .for_each(|(o, b)| {
                    for (d, s) in o.iter_mut().zip(b.chunks_exact(2)) {
                        *d = half::f16::from_bits(u16::from_le_bytes([s[0], s[1]])).to_f32();
                    }
                }),
        }
        Ok(out)
    }

    /// Like [`f32`](Self::f32) but also checks the shape.
    pub fn f32_shaped(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let info = self.info(name)?;
        ensure!(
            info.shape == shape,
            "tensor `{name}` has shape {:?}, expected {:?}",
            info.shape,
            shape
        );
        self.f32(name)
    }

    pub fn scalar(&self, name: &str) -> Result<f32> {
        let v = self.f32(name)?;
        ensure!(v.len() == 1, "tensor `{name}` is not a scalar");
        Ok(v[0])
    }

    /// Copy one row of a 2-D tensor (e.g. an embedding table) into `out` as f32.
    pub fn row_into(&self, name: &str, row: usize, out: &mut [f32]) -> Result<()> {
        let info = self.info(name)?;
        ensure!(info.shape.len() == 2, "tensor `{name}` is not 2-D");
        let (rows, cols) = (info.shape[0], info.shape[1]);
        ensure!(row < rows && out.len() == cols, "row {row} out of range for `{name}`");
        let bytes = self.bytes(info);
        match info.dtype {
            Dtype::Bf16 => {
                let b = &bytes[row * cols * 2..(row + 1) * cols * 2];
                for (d, s) in out.iter_mut().zip(b.chunks_exact(2)) {
                    *d = bf16_to_f32(u16::from_le_bytes([s[0], s[1]]));
                }
            }
            Dtype::F16 => {
                let b = &bytes[row * cols * 2..(row + 1) * cols * 2];
                for (d, s) in out.iter_mut().zip(b.chunks_exact(2)) {
                    *d = half::f16::from_bits(u16::from_le_bytes([s[0], s[1]])).to_f32();
                }
            }
            Dtype::F32 => {
                let b = &bytes[row * cols * 4..(row + 1) * cols * 4];
                for (d, s) in out.iter_mut().zip(b.chunks_exact(4)) {
                    *d = f32::from_le_bytes(s.try_into().unwrap());
                }
            }
        }
        Ok(())
    }
}

#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
