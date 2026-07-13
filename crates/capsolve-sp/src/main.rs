use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::Parser;
use image::ImageReader;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::TensorRef;

const CHARSET: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";

#[derive(Parser)]
#[command(
    name = "capsolve-sp",
    about = "High-performance CPU inference for capsolve-sp"
)]
struct Args {
    #[arg(required = true)]
    images: Vec<PathBuf>,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long)]
    runtime: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    repeat: usize,
    #[arg(long, default_value_t = 0)]
    warmup: usize,
    #[arg(long, default_value_t = 8)]
    batch_size: usize,
    #[arg(long, default_value_t = 8)]
    threads: usize,
    #[arg(long)]
    confidence: bool,
}

fn load_image(path: &Path) -> Result<Vec<f32>> {
    let image = ImageReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .with_guessed_format()?
        .decode()
        .with_context(|| format!("failed to decode {}", path.display()))?
        .to_luma8();
    if image.dimensions() != (175, 60) {
        bail!("{} must be 175x60", path.display());
    }
    Ok(image
        .into_raw()
        .into_iter()
        .map(|value| (255 - value) as f32 / 255.0)
        .collect())
}

fn make_batches(images: &[Vec<f32>], batch_size: usize) -> Vec<Vec<f32>> {
    images
        .chunks(batch_size)
        .map(|chunk| chunk.iter().flatten().copied().collect())
        .collect()
}

fn decode(values: &[f32]) -> Result<Vec<(String, [f32; 5])>> {
    let mut outputs = Vec::with_capacity(values.len() / (5 * 36));
    for sample in values.chunks_exact(5 * 36) {
        let mut text = String::with_capacity(5);
        let mut confidences = [0.0; 5];
        for (position, row) in sample.chunks_exact(36).enumerate() {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = row.iter().map(|value| (*value - max).exp()).sum();
            let (index, logit) = row
                .iter()
                .copied()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .context("empty classifier output")?;
            text.push(CHARSET[index] as char);
            confidences[position] = (logit - max).exp() / denominator;
        }
        outputs.push((text, confidences));
    }
    Ok(outputs)
}

fn run_batch(session: &mut Session, batch: &[f32]) -> Result<Vec<(String, [f32; 5])>> {
    let batch_len = batch.len() / (60 * 175);
    let input = TensorRef::from_array_view(([batch_len, 1, 60, 175], batch))?;
    let outputs = session.run(ort::inputs![input])?;
    let (_, values) = outputs[0].try_extract_tensor::<f32>()?;
    decode(values)
}

fn format_duration(duration: Duration, count: usize) -> String {
    let milliseconds = duration.as_secs_f64() * 1_000.0;
    format!(
        "{milliseconds:.3} ms total, {:.3} ms/image, {:.1} images/s",
        milliseconds / count as f64,
        count as f64 / duration.as_secs_f64()
    )
}

fn resolve_asset(explicit: Option<PathBuf>, file_name: &str) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return path
            .canonicalize()
            .with_context(|| format!("failed to locate {}", path.display()));
    }
    if let Ok(path) = PathBuf::from(file_name).canonicalize() {
        return Ok(path);
    }
    let executable = std::env::current_exe().context("failed to locate executable")?;
    for directory in executable.ancestors().skip(1).take(4) {
        let candidate = directory.join(file_name);
        if let Ok(path) = candidate.canonicalize() {
            return Ok(path);
        }
    }
    bail!("failed to locate {file_name}; pass its path explicitly")
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.repeat == 0 || args.batch_size == 0 || args.threads == 0 {
        bail!("--repeat, --batch-size, and --threads must be at least 1");
    }
    let load_start = Instant::now();
    let runtime = resolve_asset(args.runtime, "models/libonnxruntime.so")?;
    let model = resolve_asset(args.model, "models/model.onnx")?;
    ort::init_from(runtime.to_string_lossy()).commit()?;
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(args.threads)?
        .commit_from_file(model)?;
    let images = args
        .images
        .iter()
        .map(|path| load_image(path))
        .collect::<Result<Vec<_>>>()?;
    let batches = make_batches(&images, args.batch_size);
    eprintln!(
        "model+images loaded in {:.3} ms",
        load_start.elapsed().as_secs_f64() * 1_000.0
    );
    for _ in 0..args.warmup {
        for batch in &batches {
            let _ = run_batch(&mut session, batch)?;
        }
    }
    let start = Instant::now();
    let mut outputs = Vec::with_capacity(images.len());
    for iteration in 0..args.repeat {
        for batch in &batches {
            let predictions = run_batch(&mut session, batch)?;
            if iteration + 1 == args.repeat {
                outputs.extend(predictions);
            }
        }
    }
    let elapsed = start.elapsed();
    for (path, (text, confidences)) in args.images.iter().zip(outputs) {
        if args.confidence {
            println!(
                "{}: {} [{:.2}% {:.2}% {:.2}% {:.2}% {:.2}%]",
                path.display(),
                text,
                confidences[0] * 100.0,
                confidences[1] * 100.0,
                confidences[2] * 100.0,
                confidences[3] * 100.0,
                confidences[4] * 100.0,
            );
        } else {
            println!("{}: {text}", path.display());
        }
    }
    eprintln!(
        "inference: {}",
        format_duration(elapsed, images.len() * args.repeat)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_five_argmax_characters() {
        let expected = [10, 11, 12, 13, 14];
        let mut logits = vec![-10.0; 5 * 36];
        for (position, &index) in expected.iter().enumerate() {
            logits[position * 36 + index] = 10.0;
        }
        let decoded = decode(&logits).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].0, "abcde");
        assert!(decoded[0].1.iter().all(|confidence| *confidence > 0.999));
    }

    #[test]
    fn decodes_multiple_samples() {
        let mut logits = vec![0.0; 2 * 5 * 36];
        for position in 0..5 {
            logits[position * 36 + position] = 1.0;
            logits[5 * 36 + position * 36 + 35 - position] = 1.0;
        }
        let decoded = decode(&logits).unwrap();
        assert_eq!(decoded[0].0, "01234");
        assert_eq!(decoded[1].0, "zyxwv");
    }
}
