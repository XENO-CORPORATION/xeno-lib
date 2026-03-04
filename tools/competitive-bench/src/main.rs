use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use image::{imageops::FilterType, DynamicImage, GenericImageView, ImageBuffer, Rgba};
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[command(name = "competitive-bench")]
#[command(about = "Benchmark xeno against FFmpeg/ImageMagick/libvips with quality metrics")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Run(RunArgs),
    Gate(GateArgs),
    UpdateBaseline(UpdateBaselineArgs),
}

#[derive(Args, Clone)]
struct RunArgs {
    #[arg(long, default_value = "xeno-edit/target/release/xeno-edit")]
    xeno_bin: PathBuf,
    #[arg(long, default_value = "ffmpeg")]
    ffmpeg_bin: String,
    #[arg(long, default_value = "magick")]
    magick_bin: String,
    #[arg(long, default_value = "vips")]
    vips_bin: String,
    #[arg(long, default_value = "benchmarks/competitors/results/latest.json")]
    output: PathBuf,
    #[arg(long, default_value = "target/competitive-bench")]
    work_dir: PathBuf,
    #[arg(long, default_value_t = 6)]
    iterations: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 4000)]
    fixture_width: u32,
    #[arg(long, default_value_t = 2500)]
    fixture_height: u32,
    #[arg(long)]
    allow_missing_tools: bool,
}

#[derive(Args, Clone)]
struct GateArgs {
    #[arg(long, default_value = "benchmarks/competitors/results/latest.json")]
    current: PathBuf,
    #[arg(long, default_value = "benchmarks/competitors/baseline.json")]
    baseline: PathBuf,
    #[arg(long, default_value_t = 15.0)]
    max_time_regression_pct: f64,
    #[arg(long, default_value_t = 0.25)]
    max_psnr_drop: f64,
    #[arg(long, default_value_t = 0.002)]
    max_ssim_drop: f64,
    #[arg(long, default_value_t = 10.0)]
    max_slowdown_vs_best_pct: f64,
    #[arg(long, default_value_t = 12.0)]
    max_p95_slowdown_vs_best_pct: f64,
    #[arg(long, default_value_t = 0.20)]
    max_psnr_gap_vs_best: f64,
    #[arg(long, default_value_t = 0.0015)]
    max_ssim_gap_vs_best: f64,
    #[arg(long, default_value_t = 32.0)]
    min_psnr: f64,
    #[arg(long, default_value_t = 0.94)]
    min_ssim: f64,
    #[arg(long)]
    require_baseline: bool,
    #[arg(long)]
    require_competitors: bool,
    #[arg(long)]
    require_all_competitors: bool,
}

#[derive(Args)]
struct UpdateBaselineArgs {
    #[arg(long, default_value = "benchmarks/competitors/results/latest.json")]
    from: PathBuf,
    #[arg(long, default_value = "benchmarks/competitors/baseline.json")]
    to: PathBuf,
}

#[derive(Debug, Clone, Copy)]
enum Scenario {
    Resize1080,
    Blur2,
    Rotate90,
}

impl Scenario {
    fn all() -> [Scenario; 3] {
        [Scenario::Resize1080, Scenario::Blur2, Scenario::Rotate90]
    }

    fn id(self) -> &'static str {
        match self {
            Scenario::Resize1080 => "resize_4k_to_1080p",
            Scenario::Blur2 => "gaussian_blur_sigma2",
            Scenario::Rotate90 => "rotate_90",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Scenario::Resize1080 => "Resize 4000x2500 to 1920x1080",
            Scenario::Blur2 => "Gaussian blur with sigma=2.0",
            Scenario::Rotate90 => "Rotate 90 degrees clockwise",
        }
    }
}

#[derive(Debug, Clone)]
struct ToolCommand {
    tool: String,
    program: OsString,
    args: Vec<OsString>,
    output_path: PathBuf,
    availability_probe: Vec<OsString>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchReport {
    generated_at_unix_ms: u128,
    host_os: String,
    iterations: usize,
    warmup: usize,
    scenarios: Vec<ScenarioReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScenarioReport {
    id: String,
    description: String,
    reference_path: String,
    tools: Vec<ToolReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolReport {
    tool: String,
    available: bool,
    mean_ms: Option<f64>,
    p95_ms: Option<f64>,
    max_rss_kb: Option<u64>,
    quality: Option<QualityMetrics>,
    output_path: Option<String>,
    error: Option<String>,
    runs_ms: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QualityMetrics {
    psnr: f64,
    ssim: f64,
    mae: f64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => run(args),
        Commands::Gate(args) => gate(args),
        Commands::UpdateBaseline(args) => update_baseline(args),
    }
}

fn run(args: RunArgs) -> Result<()> {
    fs::create_dir_all(&args.work_dir)
        .with_context(|| format!("Failed to create work dir: {}", args.work_dir.display()))?;

    let fixture_path = args.work_dir.join("fixture_source.png");
    generate_fixture(&fixture_path, args.fixture_width, args.fixture_height)?;

    let mut scenario_reports = Vec::new();
    for scenario in Scenario::all() {
        let report = run_scenario(scenario, &fixture_path, &args)?;
        scenario_reports.push(report);
    }

    let report = BenchReport {
        generated_at_unix_ms: now_unix_ms(),
        host_os: std::env::consts::OS.to_string(),
        iterations: args.iterations,
        warmup: args.warmup,
        scenarios: scenario_reports,
    };

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output dir: {}", parent.display()))?;
    }
    fs::write(&args.output, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("Failed to write report: {}", args.output.display()))?;

    println!("Wrote benchmark report: {}", args.output.display());
    Ok(())
}

fn gate(args: GateArgs) -> Result<()> {
    let current: BenchReport =
        serde_json::from_slice(&fs::read(&args.current).with_context(|| {
            format!("Failed to read current report: {}", args.current.display())
        })?)
        .context("Failed to parse current report JSON")?;

    let baseline = if args.baseline.exists() {
        Some(
            serde_json::from_slice::<BenchReport>(&fs::read(&args.baseline).with_context(
                || {
                    format!(
                        "Failed to read baseline report: {}",
                        args.baseline.display()
                    )
                },
            )?)
            .context("Failed to parse baseline report JSON")?,
        )
    } else if args.require_baseline {
        bail!(
            "Baseline report is required but missing: {}",
            args.baseline.display()
        );
    } else {
        None
    };

    let mut failures = Vec::new();

    for scenario in &current.scenarios {
        let xeno = find_tool(scenario, "xeno");
        let Some(xeno) = xeno else {
            failures.push(format!("{}: missing xeno result", scenario.id));
            continue;
        };
        if !xeno.available {
            failures.push(format!(
                "{}: xeno unavailable ({})",
                scenario.id,
                xeno.error
                    .as_deref()
                    .unwrap_or("missing availability details")
            ));
            continue;
        }

        if let Some(quality) = &xeno.quality {
            if quality.psnr < args.min_psnr {
                failures.push(format!(
                    "{}: xeno PSNR {:.3} below minimum {:.3}",
                    scenario.id, quality.psnr, args.min_psnr
                ));
            }
            if quality.ssim < args.min_ssim {
                failures.push(format!(
                    "{}: xeno SSIM {:.6} below minimum {:.6}",
                    scenario.id, quality.ssim, args.min_ssim
                ));
            }
        } else {
            failures.push(format!("{}: missing xeno quality metrics", scenario.id));
        }

        let competitors = scenario
            .tools
            .iter()
            .filter(|t| t.tool != "xeno" && t.available)
            .collect::<Vec<_>>();

        if args.require_all_competitors {
            for competitor in scenario.tools.iter().filter(|t| t.tool != "xeno") {
                if !competitor.available {
                    failures.push(format!(
                        "{}: competitor '{}' unavailable ({})",
                        scenario.id,
                        competitor.tool,
                        competitor
                            .error
                            .as_deref()
                            .unwrap_or("missing availability details")
                    ));
                }
            }
        }

        if competitors.is_empty() && args.require_competitors {
            failures.push(format!(
                "{}: no available competitor tools (required by gate)",
                scenario.id
            ));
        }

        let best_competitor = competitors
            .iter()
            .filter_map(|t| t.mean_ms.map(|ms| (t.tool.as_str(), ms)))
            .min_by(|a, b| a.1.total_cmp(&b.1));

        if let Some((best_tool, best_ms)) = best_competitor {
            if let Some(xeno_ms) = xeno.mean_ms {
                let max_allowed = best_ms * (1.0 + args.max_slowdown_vs_best_pct / 100.0);
                if xeno_ms > max_allowed {
                    failures.push(format!(
                        "{}: xeno mean {:.3}ms slower than limit {:.3}ms (best {} {:.3}ms)",
                        scenario.id, xeno_ms, max_allowed, best_tool, best_ms
                    ));
                }
            } else {
                failures.push(format!("{}: missing xeno mean time", scenario.id));
            }
        } else if args.require_competitors {
            failures.push(format!(
                "{}: no competitor mean timings available",
                scenario.id
            ));
        }

        let best_competitor_p95 = competitors
            .iter()
            .filter_map(|t| t.p95_ms.map(|ms| (t.tool.as_str(), ms)))
            .min_by(|a, b| a.1.total_cmp(&b.1));

        if let Some((best_tool, best_p95)) = best_competitor_p95 {
            if let Some(xeno_p95) = xeno.p95_ms {
                let max_allowed = best_p95 * (1.0 + args.max_p95_slowdown_vs_best_pct / 100.0);
                if xeno_p95 > max_allowed {
                    failures.push(format!(
                        "{}: xeno p95 {:.3}ms slower than limit {:.3}ms (best {} {:.3}ms)",
                        scenario.id, xeno_p95, max_allowed, best_tool, best_p95
                    ));
                }
            } else {
                failures.push(format!("{}: missing xeno p95 time", scenario.id));
            }
        } else if args.require_competitors {
            failures.push(format!(
                "{}: no competitor p95 timings available",
                scenario.id
            ));
        }

        if let Some(xeno_quality) = &xeno.quality {
            let best_psnr = competitors
                .iter()
                .filter_map(|t| t.quality.as_ref().map(|q| (t.tool.as_str(), q.psnr)))
                .max_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((best_tool, best_psnr)) = best_psnr {
                let min_allowed = best_psnr - args.max_psnr_gap_vs_best;
                if xeno_quality.psnr < min_allowed {
                    failures.push(format!(
                        "{}: xeno PSNR {:.3} below competitor envelope {:.3} (best {} {:.3})",
                        scenario.id, xeno_quality.psnr, min_allowed, best_tool, best_psnr
                    ));
                }
            } else if args.require_competitors {
                failures.push(format!(
                    "{}: no competitor PSNR metrics available",
                    scenario.id
                ));
            }

            let best_ssim = competitors
                .iter()
                .filter_map(|t| t.quality.as_ref().map(|q| (t.tool.as_str(), q.ssim)))
                .max_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((best_tool, best_ssim)) = best_ssim {
                let min_allowed = best_ssim - args.max_ssim_gap_vs_best;
                if xeno_quality.ssim < min_allowed {
                    failures.push(format!(
                        "{}: xeno SSIM {:.6} below competitor envelope {:.6} (best {} {:.6})",
                        scenario.id, xeno_quality.ssim, min_allowed, best_tool, best_ssim
                    ));
                }
            } else if args.require_competitors {
                failures.push(format!(
                    "{}: no competitor SSIM metrics available",
                    scenario.id
                ));
            }
        }

        if let Some(ref baseline_report) = baseline {
            if baseline_report.host_os != current.host_os {
                continue;
            }
            if let Some(base_scenario) = baseline_report
                .scenarios
                .iter()
                .find(|s| s.id == scenario.id)
            {
                if let (Some(base_xeno), Some(curr_xeno)) = (
                    find_tool(base_scenario, "xeno"),
                    find_tool(scenario, "xeno"),
                ) {
                    if let (Some(base_ms), Some(curr_ms)) = (base_xeno.mean_ms, curr_xeno.mean_ms) {
                        let max_allowed = base_ms * (1.0 + args.max_time_regression_pct / 100.0);
                        if curr_ms > max_allowed {
                            failures.push(format!(
                                "{}: xeno time regression {:.3}ms > {:.3}ms (baseline {:.3}ms)",
                                scenario.id, curr_ms, max_allowed, base_ms
                            ));
                        }
                    }
                    if let (Some(base_q), Some(curr_q)) = (&base_xeno.quality, &curr_xeno.quality) {
                        if curr_q.psnr + args.max_psnr_drop < base_q.psnr {
                            failures.push(format!(
                                "{}: xeno PSNR dropped from {:.3} to {:.3}",
                                scenario.id, base_q.psnr, curr_q.psnr
                            ));
                        }
                        if curr_q.ssim + args.max_ssim_drop < base_q.ssim {
                            failures.push(format!(
                                "{}: xeno SSIM dropped from {:.6} to {:.6}",
                                scenario.id, base_q.ssim, curr_q.ssim
                            ));
                        }
                    }
                }
            }
        }
    }

    if failures.is_empty() {
        println!("Benchmark gate passed.");
        Ok(())
    } else {
        eprintln!("Benchmark gate failed:");
        for failure in failures {
            eprintln!(" - {}", failure);
        }
        bail!("Benchmark gate failed");
    }
}

fn update_baseline(args: UpdateBaselineArgs) -> Result<()> {
    if !args.from.exists() {
        bail!("Source report does not exist: {}", args.from.display());
    }
    if let Some(parent) = args.to.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create baseline dir: {}", parent.display()))?;
    }
    fs::copy(&args.from, &args.to).with_context(|| {
        format!(
            "Failed to copy {} -> {}",
            args.from.display(),
            args.to.display()
        )
    })?;
    println!("Updated baseline: {}", args.to.display());
    Ok(())
}

fn run_scenario(scenario: Scenario, fixture_path: &Path, args: &RunArgs) -> Result<ScenarioReport> {
    let scenario_dir = args.work_dir.join(scenario.id());
    let output_dir = scenario_dir.join("outputs");
    fs::create_dir_all(&output_dir).with_context(|| {
        format!(
            "Failed to create scenario output dir: {}",
            output_dir.display()
        )
    })?;

    let input_path = scenario_dir.join("input.png");
    fs::copy(fixture_path, &input_path).with_context(|| {
        format!(
            "Failed to copy fixture {} -> {}",
            fixture_path.display(),
            input_path.display()
        )
    })?;

    let reference_path = scenario_dir.join("reference.png");
    generate_reference(scenario, &input_path, &reference_path)?;

    let commands = build_tool_commands(scenario, &input_path, &output_dir, args);
    let mut tool_reports = Vec::new();
    for cmd in commands {
        let report = run_tool(
            &cmd,
            &reference_path,
            args.iterations,
            args.warmup,
            args.allow_missing_tools,
        )?;
        tool_reports.push(report);
    }

    Ok(ScenarioReport {
        id: scenario.id().to_string(),
        description: scenario.description().to_string(),
        reference_path: reference_path.to_string_lossy().to_string(),
        tools: tool_reports,
    })
}

fn run_tool(
    cmd: &ToolCommand,
    reference_path: &Path,
    iterations: usize,
    warmup: usize,
    allow_missing_tools: bool,
) -> Result<ToolReport> {
    let mut resolved = cmd.clone();
    if !is_tool_available(&resolved)?
        && resolved.tool == "imagemagick"
        && resolved.program == OsString::from("magick")
    {
        resolved.program = OsString::from("convert");
    }
    if !is_tool_available(&resolved)?
        && resolved.tool == "libvips"
        && resolved.program == OsString::from("vips")
    {
        if let Some(vips_path) = find_windows_winget_vips_binary() {
            resolved.program = vips_path.into_os_string();
        }
    }

    if !is_tool_available(&resolved)? {
        if !allow_missing_tools {
            bail!("Required tool '{}' is unavailable", cmd.tool);
        }
        return Ok(ToolReport {
            tool: cmd.tool.clone(),
            available: false,
            mean_ms: None,
            p95_ms: None,
            max_rss_kb: None,
            quality: None,
            output_path: None,
            error: Some("Tool unavailable".to_string()),
            runs_ms: Vec::new(),
        });
    }

    let mut run_times = Vec::with_capacity(iterations);
    let mut max_rss_kb: Option<u64> = None;
    for idx in 0..(warmup + iterations) {
        if resolved.output_path.exists() {
            let _ = fs::remove_file(&resolved.output_path);
        }
        let rss_file = resolved
            .output_path
            .with_extension(format!("run_{}.rss", idx));
        let (elapsed, rss) = execute_tool_command(&resolved, &rss_file)?;
        if let Some(rss_kb) = rss {
            max_rss_kb = Some(max_rss_kb.unwrap_or(0).max(rss_kb));
        }
        if idx >= warmup {
            run_times.push(elapsed.as_secs_f64() * 1000.0);
        }
    }

    if !resolved.output_path.exists() {
        return Ok(ToolReport {
            tool: cmd.tool.clone(),
            available: false,
            mean_ms: None,
            p95_ms: None,
            max_rss_kb,
            quality: None,
            output_path: None,
            error: Some(format!(
                "Tool completed but output missing: {}",
                resolved.output_path.display()
            )),
            runs_ms: run_times,
        });
    }

    let quality = Some(compute_quality(&resolved.output_path, reference_path)?);
    let (mean_ms, p95_ms) = summarize_timings(&run_times);

    Ok(ToolReport {
        tool: cmd.tool.clone(),
        available: true,
        mean_ms: Some(mean_ms),
        p95_ms: Some(p95_ms),
        max_rss_kb,
        quality,
        output_path: Some(resolved.output_path.to_string_lossy().to_string()),
        error: None,
        runs_ms: run_times,
    })
}

fn execute_tool_command(cmd: &ToolCommand, rss_file: &Path) -> Result<(Duration, Option<u64>)> {
    let use_gnu_time = cfg!(target_os = "linux") && Path::new("/usr/bin/time").exists();
    let start = Instant::now();
    let output = if use_gnu_time {
        let mut command = Command::new("/usr/bin/time");
        command
            .arg("-f")
            .arg("%M")
            .arg("-o")
            .arg(rss_file)
            .arg(&cmd.program);
        for arg in &cmd.args {
            command.arg(arg);
        }
        command
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
    } else {
        let mut command = Command::new(&cmd.program);
        for arg in &cmd.args {
            command.arg(arg);
        }
        command
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
    }
    .with_context(|| format!("Failed to start tool '{}'", cmd.tool))?;
    let elapsed = start.elapsed();

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "Tool '{}' failed (code {:?}): {}",
            cmd.tool,
            output.status.code(),
            stderr.trim()
        );
    }

    let rss = if use_gnu_time && rss_file.exists() {
        let content = fs::read_to_string(rss_file)
            .with_context(|| format!("Failed to read RSS file {}", rss_file.display()))?;
        content.trim().parse::<u64>().ok()
    } else {
        None
    };

    Ok((elapsed, rss))
}

fn is_tool_available(cmd: &ToolCommand) -> Result<bool> {
    let program_path = PathBuf::from(&cmd.program);
    if program_path.components().count() > 1 {
        return Ok(program_path.exists());
    }

    let mut command = Command::new(&cmd.program);
    for arg in &cmd.availability_probe {
        command.arg(arg);
    }
    let status = command.stdout(Stdio::null()).stderr(Stdio::null()).status();
    match status {
        Ok(s) => Ok(s.success() || s.code().is_some()),
        Err(_) => Ok(false),
    }
}

fn build_tool_commands(
    scenario: Scenario,
    input_path: &Path,
    output_dir: &Path,
    args: &RunArgs,
) -> Vec<ToolCommand> {
    let xeno_program = args.xeno_bin.as_os_str().to_os_string();
    let ffmpeg_program = OsString::from(args.ffmpeg_bin.clone());
    let magick_program = OsString::from(args.magick_bin.clone());
    let vips_program = OsString::from(args.vips_bin.clone());

    let xeno_output = output_dir.join("input_filtered.png");
    let ffmpeg_output = output_dir.join("ffmpeg.png");
    let magick_output = output_dir.join("imagemagick.png");
    let vips_output = output_dir.join("libvips.png");

    let input = input_path.as_os_str().to_os_string();
    let out_dir = output_dir.as_os_str().to_os_string();

    match scenario {
        Scenario::Resize1080 => vec![
            ToolCommand {
                tool: "xeno".to_string(),
                program: xeno_program,
                args: vec![
                    OsString::from("image-filter"),
                    input.clone(),
                    OsString::from("--output-dir"),
                    out_dir.clone(),
                    OsString::from("--width"),
                    OsString::from("1920"),
                    OsString::from("--height"),
                    OsString::from("1080"),
                    OsString::from("--quiet"),
                ],
                output_path: xeno_output,
                availability_probe: Vec::new(),
            },
            ToolCommand {
                tool: "ffmpeg".to_string(),
                program: ffmpeg_program,
                args: vec![
                    OsString::from("-hide_banner"),
                    OsString::from("-loglevel"),
                    OsString::from("error"),
                    OsString::from("-y"),
                    OsString::from("-i"),
                    input.clone(),
                    OsString::from("-vf"),
                    OsString::from("scale=1920:1080:flags=lanczos"),
                    OsString::from("-frames:v"),
                    OsString::from("1"),
                    ffmpeg_output.as_os_str().to_os_string(),
                ],
                output_path: ffmpeg_output,
                availability_probe: vec![OsString::from("-version")],
            },
            ToolCommand {
                tool: "imagemagick".to_string(),
                program: magick_program,
                args: vec![
                    input.clone(),
                    OsString::from("-filter"),
                    OsString::from("Lanczos"),
                    OsString::from("-resize"),
                    OsString::from("1920x1080!"),
                    magick_output.as_os_str().to_os_string(),
                ],
                output_path: magick_output,
                availability_probe: vec![OsString::from("-version")],
            },
            ToolCommand {
                tool: "libvips".to_string(),
                program: vips_program,
                args: vec![
                    OsString::from("resize"),
                    input.clone(),
                    vips_output.as_os_str().to_os_string(),
                    OsString::from("0.48"),
                    OsString::from("--vscale"),
                    OsString::from("0.432"),
                    OsString::from("--kernel"),
                    OsString::from("lanczos3"),
                ],
                output_path: vips_output,
                availability_probe: vec![OsString::from("--version")],
            },
        ],
        Scenario::Blur2 => vec![
            ToolCommand {
                tool: "xeno".to_string(),
                program: xeno_program,
                args: vec![
                    OsString::from("image-filter"),
                    input.clone(),
                    OsString::from("--output-dir"),
                    out_dir.clone(),
                    OsString::from("--blur"),
                    OsString::from("2"),
                    OsString::from("--quiet"),
                ],
                output_path: xeno_output,
                availability_probe: Vec::new(),
            },
            ToolCommand {
                tool: "ffmpeg".to_string(),
                program: ffmpeg_program,
                args: vec![
                    OsString::from("-hide_banner"),
                    OsString::from("-loglevel"),
                    OsString::from("error"),
                    OsString::from("-y"),
                    OsString::from("-i"),
                    input.clone(),
                    OsString::from("-vf"),
                    OsString::from("gblur=sigma=2"),
                    OsString::from("-frames:v"),
                    OsString::from("1"),
                    ffmpeg_output.as_os_str().to_os_string(),
                ],
                output_path: ffmpeg_output,
                availability_probe: vec![OsString::from("-version")],
            },
            ToolCommand {
                tool: "imagemagick".to_string(),
                program: magick_program,
                args: vec![
                    input.clone(),
                    OsString::from("-gaussian-blur"),
                    OsString::from("0x2"),
                    magick_output.as_os_str().to_os_string(),
                ],
                output_path: magick_output,
                availability_probe: vec![OsString::from("-version")],
            },
            ToolCommand {
                tool: "libvips".to_string(),
                program: vips_program,
                args: vec![
                    OsString::from("gaussblur"),
                    input.clone(),
                    vips_output.as_os_str().to_os_string(),
                    OsString::from("2"),
                ],
                output_path: vips_output,
                availability_probe: vec![OsString::from("--version")],
            },
        ],
        Scenario::Rotate90 => vec![
            ToolCommand {
                tool: "xeno".to_string(),
                program: xeno_program,
                args: vec![
                    OsString::from("image-filter"),
                    input.clone(),
                    OsString::from("--output-dir"),
                    out_dir.clone(),
                    OsString::from("--rotate"),
                    OsString::from("90"),
                    OsString::from("--quiet"),
                ],
                output_path: xeno_output,
                availability_probe: Vec::new(),
            },
            ToolCommand {
                tool: "ffmpeg".to_string(),
                program: ffmpeg_program,
                args: vec![
                    OsString::from("-hide_banner"),
                    OsString::from("-loglevel"),
                    OsString::from("error"),
                    OsString::from("-y"),
                    OsString::from("-i"),
                    input.clone(),
                    OsString::from("-vf"),
                    OsString::from("transpose=1"),
                    OsString::from("-frames:v"),
                    OsString::from("1"),
                    ffmpeg_output.as_os_str().to_os_string(),
                ],
                output_path: ffmpeg_output,
                availability_probe: vec![OsString::from("-version")],
            },
            ToolCommand {
                tool: "imagemagick".to_string(),
                program: magick_program,
                args: vec![
                    input.clone(),
                    OsString::from("-rotate"),
                    OsString::from("90"),
                    magick_output.as_os_str().to_os_string(),
                ],
                output_path: magick_output,
                availability_probe: vec![OsString::from("-version")],
            },
            ToolCommand {
                tool: "libvips".to_string(),
                program: vips_program,
                args: vec![
                    OsString::from("rot"),
                    input.clone(),
                    vips_output.as_os_str().to_os_string(),
                    OsString::from("d90"),
                ],
                output_path: vips_output,
                availability_probe: vec![OsString::from("--version")],
            },
        ],
    }
}

fn generate_fixture(path: &Path, width: u32, height: u32) -> Result<()> {
    if path.exists() {
        return Ok(());
    }

    let image = ImageBuffer::from_fn(width, height, |x, y| {
        let nx = x as f32 / width.max(1) as f32;
        let ny = y as f32 / height.max(1) as f32;
        let r = ((nx * 255.0) + ((y % 251) as f32 * 0.3))
            .round()
            .clamp(0.0, 255.0) as u8;
        let g = ((ny * 255.0) + ((x % 233) as f32 * 0.25))
            .round()
            .clamp(0.0, 255.0) as u8;
        let b = ((((nx + ny) * 0.5) * 255.0) + (((x ^ y) % 199) as f32 * 0.2))
            .round()
            .clamp(0.0, 255.0) as u8;
        Rgba([r, g, b, 255])
    });
    DynamicImage::ImageRgba8(image)
        .save(path)
        .with_context(|| format!("Failed to save fixture: {}", path.display()))?;
    Ok(())
}

fn generate_reference(scenario: Scenario, input_path: &Path, output_path: &Path) -> Result<()> {
    let input = image::open(input_path)
        .with_context(|| format!("Failed to open scenario input: {}", input_path.display()))?;

    let out = match scenario {
        Scenario::Resize1080 => input.resize_exact(1920, 1080, FilterType::Lanczos3),
        Scenario::Blur2 => input.blur(2.0),
        Scenario::Rotate90 => input.rotate90(),
    };

    out.save(output_path)
        .with_context(|| format!("Failed to save reference output: {}", output_path.display()))?;
    Ok(())
}

fn compute_quality(output_path: &Path, reference_path: &Path) -> Result<QualityMetrics> {
    let output = image::open(output_path)
        .with_context(|| format!("Failed to open output image: {}", output_path.display()))?;
    let reference = image::open(reference_path).with_context(|| {
        format!(
            "Failed to open reference image: {}",
            reference_path.display()
        )
    })?;

    let (ow, oh) = output.dimensions();
    let (rw, rh) = reference.dimensions();
    if ow != rw || oh != rh {
        bail!(
            "Dimension mismatch output={}x{}, reference={}x{}",
            ow,
            oh,
            rw,
            rh
        );
    }

    let out = output.to_rgba8();
    let refi = reference.to_rgba8();

    let mut mae_sum = 0.0f64;
    let mut mse_sum = 0.0f64;
    let mut lum_out = Vec::with_capacity((ow * oh) as usize);
    let mut lum_ref = Vec::with_capacity((ow * oh) as usize);

    for (a, b) in out.pixels().zip(refi.pixels()) {
        for c in 0..4 {
            let da = a.0[c] as f64;
            let db = b.0[c] as f64;
            let diff = (da - db).abs();
            mae_sum += diff;
            mse_sum += diff * diff;
        }

        let la = 0.299 * a.0[0] as f64 + 0.587 * a.0[1] as f64 + 0.114 * a.0[2] as f64;
        let lb = 0.299 * b.0[0] as f64 + 0.587 * b.0[1] as f64 + 0.114 * b.0[2] as f64;
        lum_out.push(la);
        lum_ref.push(lb);
    }

    let n = (ow as usize * oh as usize * 4) as f64;
    let mae = mae_sum / n;
    let mse = mse_sum / n;
    let psnr = if mse <= f64::EPSILON {
        100.0
    } else {
        20.0 * 255.0f64.log10() - 10.0 * mse.log10()
    };
    let ssim = compute_global_ssim(&lum_out, &lum_ref);

    Ok(QualityMetrics { psnr, ssim, mae })
}

fn compute_global_ssim(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;

    let mut var_a = 0.0;
    let mut var_b = 0.0;
    let mut cov = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        var_a += dx * dx;
        var_b += dy * dy;
        cov += dx * dy;
    }

    let denom = (n - 1.0).max(1.0);
    var_a /= denom;
    var_b /= denom;
    cov /= denom;

    let c1 = (0.01f64 * 255.0f64).powi(2);
    let c2 = (0.03f64 * 255.0f64).powi(2);
    let num = (2.0 * mean_a * mean_b + c1) * (2.0 * cov + c2);
    let den = (mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2);
    if den.abs() <= f64::EPSILON {
        1.0
    } else {
        (num / den).clamp(-1.0, 1.0)
    }
}

fn summarize_timings(samples: &[f64]) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let p95_index = ((sorted.len() - 1) as f64 * 0.95).round() as usize;
    let p95 = sorted[p95_index.min(sorted.len() - 1)];
    (mean, p95)
}

fn find_tool<'a>(scenario: &'a ScenarioReport, tool: &str) -> Option<&'a ToolReport> {
    scenario.tools.iter().find(|t| t.tool == tool)
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn find_windows_winget_vips_binary() -> Option<PathBuf> {
    if !cfg!(target_os = "windows") {
        return None;
    }

    let local_app_data = std::env::var_os("LOCALAPPDATA")?;
    let winget_packages = PathBuf::from(local_app_data)
        .join("Microsoft")
        .join("WinGet")
        .join("Packages");
    if !winget_packages.exists() {
        return None;
    }

    let mut candidates = Vec::new();
    for package_dir in fs::read_dir(winget_packages).ok()? {
        let package_dir = package_dir.ok()?;
        let package_name = package_dir.file_name().to_string_lossy().to_string();
        if !package_name.starts_with("libvips.libvips_") {
            continue;
        }

        for version_dir in fs::read_dir(package_dir.path()).ok()? {
            let version_dir = version_dir.ok()?;
            let candidate = version_dir.path().join("bin").join("vips.exe");
            if candidate.exists() {
                candidates.push(candidate);
            }
        }
    }

    candidates.sort();
    candidates.pop()
}

#[allow(dead_code)]
fn _os_str<S: AsRef<OsStr>>(s: S) -> OsString {
    s.as_ref().to_os_string()
}
