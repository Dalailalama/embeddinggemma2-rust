//! Image / video-frame preprocessing for the vision tower.
//!
//! Mirrors `Gemma4ImageProcessorPil`: aspect-ratio-preserving resize to a patch budget using
//! Pillow's bicubic resampling (re-implemented here bit-for-bit, fixed-point and all), rescale to
//! `[0, 1]`, then cut into `patch x patch x 3` patches with 2-D positions.

use anyhow::{Result, bail};
use rayon::prelude::*;

/// Patches of one image, in row-major patch order.
pub struct PatchedImage {
    /// `[n_patches, patch*patch*3]`, each patch flattened as `(row, col, channel)`.
    pub patches: Vec<f32>,
    /// `(x, y)` patch coordinates.
    pub positions: Vec<[u32; 2]>,
    pub grid_h: usize,
    pub grid_w: usize,
    pub patch_dim: usize,
}

impl PatchedImage {
    pub fn n_patches(&self) -> usize {
        self.grid_h * self.grid_w
    }

    pub fn num_soft_tokens(&self, pooling_kernel: usize) -> usize {
        self.n_patches() / (pooling_kernel * pooling_kernel)
    }
}

/// Largest `(height, width)` that fits `max_patches` patches, is a multiple of
/// `pooling_kernel_size * patch_size`, and keeps the aspect ratio.
pub fn aspect_ratio_preserving_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> Result<(usize, usize)> {
    let total_px = (height * width) as f64;
    let target_px = (max_patches * patch_size * patch_size) as f64;
    let factor = (target_px / total_px).sqrt();
    let ideal_h = factor * height as f64;
    let ideal_w = factor * width as f64;
    let side_mult = pooling_kernel_size * patch_size;

    let mut target_h = (ideal_h / side_mult as f64).floor() as usize * side_mult;
    let mut target_w = (ideal_w / side_mult as f64).floor() as usize * side_mult;
    if target_h == 0 && target_w == 0 {
        bail!("attempting to resize to a 0 x 0 image (side multiple is {side_mult})");
    }
    let max_side = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
    if target_h == 0 {
        target_h = side_mult;
        target_w = ((width as f64 / height as f64).floor() as usize * side_mult).min(max_side);
    } else if target_w == 0 {
        target_w = side_mult;
        target_h = ((height as f64 / width as f64).floor() as usize * side_mult).min(max_side);
    }
    if (target_h * target_w) as f64 > target_px {
        bail!("resizing {height}x{width} to {target_h}x{target_w} exceeds {max_patches} patches");
    }
    Ok((target_h, target_w))
}

// ---------------------------------------------------------------------------------------------
// Pillow-compatible bicubic resampling (8 bits per channel)
// ---------------------------------------------------------------------------------------------

/// Which fixed-point resampler to emulate. Both share Pillow's filter geometry and two-pass
/// (horizontal, then vertical) structure and differ only in weight precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResizeMode {
    /// PyTorch's uint8 antialiased bicubic (`torchvision.transforms.v2.functional.resize`), used by
    /// the reference image/video processors: int16 weights with a data-dependent precision.
    #[default]
    Torch,
    /// Pillow's `Image.resize(..., BICUBIC)`: int32 weights with 22 fractional bits.
    Pillow,
}

const PILLOW_PRECISION_BITS: u32 = 32 - 8 - 2;

fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

struct Coeffs {
    precision: u32,
    ksize: usize,
    /// `(xmin, len)` of the contributing input range for every output index.
    bounds: Vec<(usize, usize)>,
    /// Fixed-point weights, `ksize` per output index.
    kk: Vec<i32>,
}

fn precompute_coeffs(in_size: usize, out_size: usize, mode: ResizeMode) -> Coeffs {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut kf_all = vec![0f64; out_size * ksize];
    let mut bounds = Vec::with_capacity(out_size);
    let mut kf = vec![0f64; ksize];
    let mut wt_max = 0f64;
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = (((center + support + 0.5) as i64).min(in_size as i64)).max(0) as usize;
        let len = xmax.saturating_sub(xmin);
        let mut ww = 0.0;
        for (x, k) in kf.iter_mut().enumerate().take(len) {
            let w = bicubic((x as f64 + xmin as f64 - center + 0.5) * ss);
            *k = w;
            ww += w;
        }
        for (x, k) in kf.iter_mut().enumerate() {
            if x < len {
                if ww != 0.0 {
                    *k /= ww;
                }
            } else {
                *k = 0.0;
            }
        }
        for (x, k) in kf.iter().enumerate() {
            kf_all[xx * ksize + x] = *k;
            wt_max = wt_max.max(*k);
        }
        bounds.push((xmin, len));
    }
    let precision = match mode {
        ResizeMode::Pillow => PILLOW_PRECISION_BITS,
        ResizeMode::Torch => {
            // Smallest p such that the largest weight, scaled by 2^(p+1), reaches 2^15 (fits an i16).
            let mut p = 0u32;
            while p < 22 {
                if (0.5 + wt_max * (1u64 << (p + 1)) as f64) as i64 >= (1 << 15) {
                    break;
                }
                p += 1;
            }
            p
        }
    };
    let scale_w = (1u64 << precision) as f64;
    // C's `(int)(+-0.5 + x)` truncates toward zero.
    let kk = kf_all
        .iter()
        .map(|k| {
            let v = *k * scale_w;
            if v < 0.0 { (-0.5 + v) as i32 } else { (0.5 + v) as i32 }
        })
        .collect();
    Coeffs { precision, ksize, bounds, kk }
}

#[inline]
fn clip8(v: i64, precision: u32) -> u8 {
    (v >> precision).clamp(0, 255) as u8
}

/// Resize an interleaved RGB8 image with antialiased bicubic resampling (see [`ResizeMode`]).
pub fn resize_bicubic_rgb(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, mode: ResizeMode) -> Vec<u8> {
    assert_eq!(src.len(), sw * sh * 3);
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let need_h = dw != sw;
    let need_v = dh != sh;
    let vert = precompute_coeffs(sh, dh, mode);

    // Rows of the (horizontally resized) intermediate image the vertical pass will read.
    let (first, last) = if need_v {
        let first = vert.bounds[0].0;
        let (lx, ll) = vert.bounds[dh - 1];
        (first, lx + ll)
    } else {
        (0, sh)
    };

    let tmp_w = if need_h { dw } else { sw };
    let tmp: Vec<u8> = if need_h {
        let hc = precompute_coeffs(sw, dw, mode);
        let mut tmp = vec![0u8; (last - first) * dw * 3];
        tmp.par_chunks_mut(dw * 3).enumerate().for_each(|(r, out_row)| {
            let in_row = &src[(first + r) * sw * 3..(first + r + 1) * sw * 3];
            for xx in 0..dw {
                let (xmin, len) = hc.bounds[xx];
                let k = &hc.kk[xx * hc.ksize..xx * hc.ksize + len];
                let mut acc = [1i64 << (hc.precision - 1); 3];
                for (x, kv) in k.iter().enumerate() {
                    let px = &in_row[(xmin + x) * 3..(xmin + x) * 3 + 3];
                    acc[0] += px[0] as i64 * *kv as i64;
                    acc[1] += px[1] as i64 * *kv as i64;
                    acc[2] += px[2] as i64 * *kv as i64;
                }
                out_row[xx * 3] = clip8(acc[0], hc.precision);
                out_row[xx * 3 + 1] = clip8(acc[1], hc.precision);
                out_row[xx * 3 + 2] = clip8(acc[2], hc.precision);
            }
        });
        tmp
    } else {
        src[first * sw * 3..last * sw * 3].to_vec()
    };

    if !need_v {
        return tmp;
    }
    let mut out = vec![0u8; dh * dw * 3];
    out.par_chunks_mut(dw * 3).enumerate().for_each(|(yy, out_row)| {
        let (ymin, len) = vert.bounds[yy];
        let k = &vert.kk[yy * vert.ksize..yy * vert.ksize + len];
        for xx in 0..tmp_w * 3 {
            let mut acc = 1i64 << (vert.precision - 1);
            for (y, kv) in k.iter().enumerate() {
                acc += tmp[(ymin + y - first) * tmp_w * 3 + xx] as i64 * *kv as i64;
            }
            out_row[xx] = clip8(acc, vert.precision);
        }
    });
    out
}

// ---------------------------------------------------------------------------------------------
// Patchify
// ---------------------------------------------------------------------------------------------

/// Resize to the patch budget and cut into patches. `rgb` is interleaved RGB8.
pub fn preprocess_rgb(
    rgb: &[u8],
    height: usize,
    width: usize,
    patch_size: usize,
    pooling_kernel_size: usize,
    max_soft_tokens: usize,
    mode: ResizeMode,
) -> Result<PatchedImage> {
    let max_patches = max_soft_tokens * pooling_kernel_size * pooling_kernel_size;
    let (th, tw) = aspect_ratio_preserving_size(height, width, patch_size, max_patches, pooling_kernel_size)?;
    let resized = resize_bicubic_rgb(rgb, width, height, tw, th, mode);
    Ok(patchify(&resized, th, tw, patch_size))
}

/// Cut an RGB8 image (`h`, `w` multiples of `patch`) into flattened `[0,1]` patches.
pub fn patchify(rgb: &[u8], h: usize, w: usize, patch: usize) -> PatchedImage {
    let (gh, gw) = (h / patch, w / patch);
    let patch_dim = patch * patch * 3;
    // `do_rescale` multiplies by the f64 factor 1/255 in NumPy, then casts to float32.
    const RESCALE: f64 = 0.00392156862745098;
    let mut patches = vec![0f32; gh * gw * patch_dim];
    patches.par_chunks_mut(patch_dim).enumerate().for_each(|(idx, out)| {
        let (ph, pw) = (idx / gw, idx % gw);
        for py in 0..patch {
            let row = (ph * patch + py) * w * 3 + pw * patch * 3;
            for i in 0..patch * 3 {
                out[py * patch * 3 + i] = (rgb[row + i] as f64 * RESCALE) as f32;
            }
        }
    });
    let mut positions = Vec::with_capacity(gh * gw);
    for y in 0..gh {
        for x in 0..gw {
            positions.push([x as u32, y as u32]);
        }
    }
    PatchedImage { patches, positions, grid_h: gh, grid_w: gw, patch_dim }
}
