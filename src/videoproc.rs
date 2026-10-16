//! Video frame sampling, mirroring `EmbeddingGemma2VideoProcessor.sample_frames`.

use anyhow::{Result, bail, ensure};

#[derive(Debug, Clone)]
pub struct SampleParams {
    /// Target sampling rate in frames per second (`None` keeps every frame).
    pub fps: Option<f64>,
    pub max_frames: usize,
    /// `"uniform"` or `"truncate"`.
    pub overflow_strategy: String,
}

/// Choose which of `total_frames` decoded frames to keep.
///
/// `source_fps` is the native frame rate; without a usable one, rate-based sampling cannot be
/// applied (the reference warns and keeps every frame, applying only the `max_frames` budget).
/// Below the target rate the rule repeats frames, exactly like the reference.
pub fn sample_frames(total_frames: usize, source_fps: Option<f64>, p: &SampleParams) -> Result<Vec<usize>> {
    if total_frames == 0 {
        bail!("video has no frames");
    }
    ensure!(p.max_frames > 0, "max_frames must be at least 1");
    if let Some(f) = p.fps {
        ensure!(f.is_finite() && f > 0.0, "target fps must be a positive number (got {f})");
    }
    let source_fps = source_fps.filter(|f| f.is_finite() && *f > 0.0);

    // Step 1 conceptually builds `n` indices `min(total - 1, int(i * step))`; step 2 keeps at most
    // `max_frames` of them. The list is never materialised, so extreme rates cannot exhaust memory.
    let (n, step) = match (p.fps, source_fps) {
        (Some(target), Some(src)) => {
            let duration = total_frames as f64 / src;
            (((duration * target) as usize).max(1), Some(src / target))
        }
        _ => (total_frames, None),
    };
    let index = |i: usize| match step {
        Some(s) => ((i as f64 * s) as usize).min(total_frames - 1),
        None => i,
    };
    let positions: Vec<usize> = if n <= p.max_frames {
        (0..n).collect()
    } else {
        let m = p.max_frames;
        match p.overflow_strategy.as_str() {
            "truncate" => (0..m).collect(),
            // np.linspace(0, n - 1, m, dtype=int): truncation toward zero, exact endpoint.
            "uniform" if m == 1 => vec![0],
            "uniform" => {
                let lin = (n - 1) as f64 / (m - 1) as f64;
                (0..m).map(|i| if i == m - 1 { n - 1 } else { (i as f64 * lin) as usize }).collect()
            }
            other => bail!("unknown overflow_strategy `{other}` (expected `uniform` or `truncate`)"),
        }
    };
    Ok(positions.into_iter().map(index).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(fps: Option<f64>) -> SampleParams {
        SampleParams { fps, max_frames: 32, overflow_strategy: "uniform".into() }
    }

    #[test]
    fn absurd_rates_stay_bounded() {
        // A tiny source rate would ask for ~1e300 indices; only max_frames are produced.
        let idx = sample_frames(5, Some(1e-300), &params(Some(1.0))).unwrap();
        assert_eq!(idx, vec![0; 32]);
        let idx = sample_frames(1, Some(1e-6), &params(Some(1.0))).unwrap();
        assert_eq!(idx.len(), 32);
        assert!(idx.iter().all(|&i| i == 0));
        assert!(sample_frames(10, Some(30.0), &params(Some(f64::NAN))).is_err());
    }

    #[test]
    fn below_target_rate_repeats_frames() {
        // 12 frames at 0.5 fps = 24 s -> 24 one-per-second samples, each frame twice.
        let idx = sample_frames(12, Some(0.5), &params(Some(1.0))).unwrap();
        assert_eq!(idx.len(), 24);
        assert_eq!(&idx[..6], &[0, 0, 1, 1, 2, 2]);
    }
}
