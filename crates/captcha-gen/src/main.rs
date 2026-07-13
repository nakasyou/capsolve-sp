use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use fontdue::{Font, FontSettings};
use image::imageops::FilterType;
use image::{DynamicImage, GrayImage, Luma};
use rand::distr::{Distribution, Uniform};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Serialize;

const WIDTH: u32 = 175;
const HEIGHT: u32 = 60;
const FONT_SIZE: f32 = 28.0;
const DOT_NOISE: usize = 100;
const LINE_NOISE: usize = 3;

#[derive(Parser)]
#[command(
    name = "captcha-gen",
    version,
    about = "Fast standalone CAPTCHA generator"
)]
struct Cli {
    #[arg(long, global = true, help = "TrueType/OpenType font path")]
    font: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate exactly one CAPTCHA.
    One {
        /// Exact label to render (must contain five characters).
        label: String,
        /// Output path; format is selected from its extension.
        output: PathBuf,
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Generate random CAPTCHAs, forever unless --count is supplied.
    Random {
        /// Output pattern. Supports {prefix}, {label}, {id}, and {ext}.
        pattern: String,
        #[arg(long, default_value = "0123456789abcdefghijklmnopqrstuvwxyz")]
        charset: String,
        #[arg(long, default_value_t = 5)]
        length: usize,
        /// Number to generate. Omit to run until interrupted.
        #[arg(long)]
        count: Option<u64>,
        #[arg(long, default_value_t = 0)]
        start_id: u64,
        #[arg(long, default_value = "gif")]
        ext: String,
        #[arg(long)]
        seed: Option<u64>,
        /// Generator threads. Use 0 to select the available CPU count.
        #[arg(long, default_value_t = 0)]
        workers: usize,
        /// JSON Lines metadata file to append after images are saved.
        #[arg(long, default_value = "dataset/metadata.jsonl")]
        metadata: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let font_path = match cli.font {
        Some(path) => path,
        None => find_font()?,
    };
    let font_bytes = fs::read(&font_path)
        .with_context(|| format!("failed to read font {}", font_path.display()))?;
    let font = Font::from_bytes(font_bytes, FontSettings::default())
        .map_err(|error| anyhow::anyhow!("failed to parse font: {error}"))?;

    match cli.command {
        Command::One {
            label,
            output,
            seed,
        } => {
            validate_label(&label, 5)?;
            let mut rng = seeded_rng(seed);
            save_captcha(&font, &label, &output, &mut rng)?;
            println!("{}", output.display());
        }
        Command::Random {
            pattern,
            charset,
            length,
            count,
            start_id,
            ext,
            seed,
            workers,
            metadata,
        } => {
            if charset.is_empty() {
                bail!("--charset must not be empty");
            }
            if length != 5 {
                bail!("--length must be 5 for the current CAPTCHA layout");
            }
            if !pattern.contains("{label}") && !pattern.contains("{id}") {
                bail!("pattern must contain {{label}} or {{id}} to avoid overwriting files");
            }
            let chars: Vec<char> = charset.chars().collect();
            let mut seed_rng = seeded_rng(seed);
            let threads = if workers == 0 {
                std::thread::available_parallelism().map_or(1, usize::from)
            } else {
                workers
            };
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .context("failed to create generator thread pool")?;
            let batch_size = threads.saturating_mul(64).max(1) as u64;
            if let Some(parent) = metadata
                .parent()
                .filter(|path| !path.as_os_str().is_empty())
            {
                fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create {}", parent.display()))?;
            }
            let metadata_file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&metadata)
                .with_context(|| format!("failed to open {}", metadata.display()))?;
            let mut metadata_writer = BufWriter::new(metadata_file);
            let metadata_root = metadata.parent().unwrap_or_else(|| Path::new(""));
            let mut generated = 0_u64;
            while count.is_none_or(|limit| generated < limit) {
                let remaining = count.map_or(batch_size, |limit| limit - generated);
                let current_size = remaining.min(batch_size);
                let jobs: Vec<(u64, u64)> = (0..current_size)
                    .map(|offset| {
                        let id = start_id
                            .checked_add(generated + offset)
                            .context("id overflow")?;
                        Ok((id, seed_rng.random()))
                    })
                    .collect::<Result<_>>()?;
                let records: Vec<Metadata> = pool.install(|| {
                    jobs.par_iter()
                        .map(|&(id, job_seed)| {
                            let mut rng = ChaCha8Rng::seed_from_u64(job_seed);
                            let indexes = Uniform::new(0, chars.len()).expect("non-empty charset");
                            let label: String = (0..length)
                                .map(|_| chars[indexes.sample(&mut rng)])
                                .collect();
                            let path = expand_pattern(&pattern, &label, id, &ext);
                            save_captcha(&font, &label, Path::new(&path), &mut rng)?;
                            let file_name = Path::new(&path)
                                .strip_prefix(metadata_root)
                                .unwrap_or_else(|_| Path::new(&path))
                                .to_string_lossy()
                                .replace('\\', "/");
                            Ok(Metadata {
                                file_name,
                                text: label,
                            })
                        })
                        .collect::<Result<_>>()
                })?;
                for record in records {
                    serde_json::to_writer(&mut metadata_writer, &record)?;
                    metadata_writer.write_all(b"\n")?;
                }
                metadata_writer.flush()?;
                generated += current_size;
                if generated % 1_000 < current_size || count == Some(generated) {
                    eprintln!("generated {generated}");
                }
            }
            eprintln!("generated {generated} total");
        }
    }
    Ok(())
}

#[derive(Serialize)]
struct Metadata {
    file_name: String,
    text: String,
}

fn seeded_rng(seed: Option<u64>) -> ChaCha8Rng {
    match seed {
        Some(seed) => ChaCha8Rng::seed_from_u64(seed),
        None => ChaCha8Rng::from_os_rng(),
    }
}

fn validate_label(label: &str, length: usize) -> Result<()> {
    if label.chars().count() != length {
        bail!("label must contain exactly {length} characters");
    }
    Ok(())
}

fn find_font() -> Result<PathBuf> {
    const CANDIDATES: &[&str] = &[
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ];
    if let Some(path) = CANDIDATES.iter().map(PathBuf::from).find(|p| p.is_file()) {
        return Ok(path);
    }

    if let Ok(output) = ProcessCommand::new("fc-match")
        .args(["-f", "%{file}\\n", "DejaVu Sans:style=Bold"])
        .output()
        && output.status.success()
        && let Some(line) = String::from_utf8_lossy(&output.stdout).lines().next()
    {
        let path = PathBuf::from(line);
        if path.is_file() {
            return Ok(path);
        }
    }

    bail!("no suitable bold font found; pass one explicitly with --font /path/to/font.ttf")
}

fn expand_pattern(pattern: &str, label: &str, id: u64, ext: &str) -> String {
    let prefix: String = label.chars().take(2).collect();
    pattern
        .replace("{prefix}", &prefix)
        .replace("{label}", label)
        .replace("{id}", &id.to_string())
        .replace("{ext}", ext.trim_start_matches('.'))
}

fn save_captcha(font: &Font, label: &str, output: &Path, rng: &mut ChaCha8Rng) -> Result<()> {
    validate_label(label, 5)?;
    let image = make_captcha(font, label, rng);
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let encoded = if output
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("gif"))
    {
        DynamicImage::ImageRgb8(DynamicImage::ImageLuma8(image).to_rgb8())
    } else {
        DynamicImage::ImageLuma8(image)
    };
    encoded
        .save(output)
        .with_context(|| format!("failed to save {}", output.display()))
}

fn make_captcha(font: &Font, text: &str, rng: &mut ChaCha8Rng) -> GrayImage {
    let mut result = GrayImage::from_pixel(WIDTH, HEIGHT, Luma([255]));
    let cell_width = 34.0_f32;
    let left = (WIDTH as f32 - cell_width * 5.0) / 2.0;

    for (index, character) in text.chars().enumerate() {
        let mut layer = GrayImage::from_pixel(WIDTH, HEIGHT, Luma([255]));
        let (metrics, bitmap) = font.rasterize(character, FONT_SIZE);
        let jitter = rng.random_range(-3.0_f32..3.0);
        let x =
            left + index as f32 * cell_width + (cell_width - metrics.width as f32) / 2.0 + jitter;
        let baseline = (HEIGHT as f32 + FONT_SIZE) / 2.0;
        let y = baseline - metrics.height as f32 - metrics.ymin as f32;
        for by in 0..metrics.height {
            for bx in 0..metrics.width {
                let px = x.round() as i32 + bx as i32;
                let py = y.round() as i32 + by as i32;
                if px >= 0 && py >= 0 && px < WIDTH as i32 && py < HEIGHT as i32 {
                    let coverage = bitmap[by * metrics.width + bx];
                    layer.put_pixel(px as u32, py as u32, Luma([255 - coverage]));
                }
            }
        }
        let warped = wave(&layer, rng);
        darken(&mut result, &warped);
    }

    noise(&mut result, rng, 2, true);
    noise(&mut result, rng, 1, false);
    result
}

fn wave(source: &GrayImage, rng: &mut ChaCha8Rng) -> GrayImage {
    let freq: [f32; 4] =
        std::array::from_fn(|_| rng.random_range(700_000..=1_000_000) as f32 / 15_000_000.0);
    let phase: [f32; 4] =
        std::array::from_fn(|_| rng.random_range(0..=3_141_592) as f32 / 1_000_000.0);
    let size_x = rng.random_range(300..=700) as f32 / 100.0;
    let size_y = rng.random_range(300..=700) as f32 / 100.0;
    let mut output = GrayImage::from_pixel(WIDTH, HEIGHT, Luma([255]));

    for x in 0..WIDTH {
        for y in 0..HEIGHT {
            let xf = x as f32;
            let yf = y as f32;
            let sx =
                xf + ((xf * freq[0] + phase[0]).sin() + (yf * freq[2] + phase[2]).sin()) * size_x;
            let sy =
                yf + ((xf * freq[1] + phase[1]).sin() + (yf * freq[3] + phase[3]).sin()) * size_y;
            if sx < 0.0 || sy < 0.0 || sx >= (WIDTH - 1) as f32 || sy >= (HEIGHT - 1) as f32 {
                continue;
            }
            let ix = sx.floor() as u32;
            let iy = sy.floor() as u32;
            let fx = sx - ix as f32;
            let fy = sy - iy as f32;
            let c = source.get_pixel(ix, iy)[0] as f32;
            let cx = source.get_pixel(ix + 1, iy)[0] as f32;
            let cy = source.get_pixel(ix, iy + 1)[0] as f32;
            let cxy = source.get_pixel(ix + 1, iy + 1)[0] as f32;
            let value = c * (1.0 - fx) * (1.0 - fy)
                + cx * fx * (1.0 - fy)
                + cy * (1.0 - fx) * fy
                + cxy * fx * fy;
            output.put_pixel(x, y, Luma([value.round() as u8]));
        }
    }
    output
}

fn noise(image: &mut GrayImage, rng: &mut ChaCha8Rng, dot_size: u32, lines: bool) {
    const SCALE: u32 = 4;
    let sw = WIDTH * SCALE;
    let sh = HEIGHT * SCALE;
    let mut overlay = GrayImage::from_pixel(sw, sh, Luma([255]));
    let radius = (dot_size * SCALE) as i32 / 2;
    for _ in 0..DOT_NOISE {
        let cx = rng.random_range(0..=sw) as i32;
        let cy = rng.random_range(0..=sh) as i32;
        fill_circle(&mut overlay, cx, cy, radius);
    }
    if lines {
        for _ in 0..LINE_NOISE {
            let end = (
                rng.random_range(0..=sw) as i32,
                rng.random_range(0..=sh) as i32,
            );
            let mut points = vec![(0, 0)];
            if rng.random_bool(0.5) {
                points.push((
                    rng.random_range(0..=sw) as i32,
                    rng.random_range(0..=sh) as i32,
                ));
            }
            points.push(end);
            for pair in points.windows(2) {
                draw_thick_line(&mut overlay, pair[0], pair[1], SCALE as i32);
            }
        }
    }
    let filter = if dot_size == 1 {
        FilterType::Lanczos3
    } else {
        FilterType::Lanczos3
    };
    let resized = image::imageops::resize(&overlay, WIDTH, HEIGHT, filter);
    darken(image, &resized);
}

fn fill_circle(image: &mut GrayImage, cx: i32, cy: i32, radius: i32) {
    for y in (cy - radius)..=(cy + radius) {
        for x in (cx - radius)..=(cx + radius) {
            if x >= 0
                && y >= 0
                && x < image.width() as i32
                && y < image.height() as i32
                && (x - cx).pow(2) + (y - cy).pow(2) <= radius.pow(2)
            {
                image.put_pixel(x as u32, y as u32, Luma([0]));
            }
        }
    }
}

fn draw_thick_line(image: &mut GrayImage, start: (i32, i32), end: (i32, i32), width: i32) {
    let dx = end.0 - start.0;
    let dy = end.1 - start.1;
    let steps = dx.abs().max(dy.abs()).max(1);
    for step in 0..=steps {
        let x = start.0 + dx * step / steps;
        let y = start.1 + dy * step / steps;
        fill_circle(image, x, y, width / 2);
    }
}

fn darken(target: &mut GrayImage, source: &GrayImage) {
    for (target_pixel, source_pixel) in target.pixels_mut().zip(source.pixels()) {
        target_pixel[0] = target_pixel[0].min(source_pixel[0]);
    }
}
