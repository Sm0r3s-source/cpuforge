mod args;
mod cmd_bench;
mod cmd_demo;
mod cmd_generate;
mod cmd_init;
mod cmd_merge;
mod cmd_parity;
mod cmd_pull;
mod cmd_train;
mod cmd_update;
mod cmd_worker;
mod cmd_ui;
mod status;
mod util;

use args::Args;

const USAGE: &str = "\
cpuforge - CPU-only finetuning for Llama-style models

USAGE: cpuforge <command> [--flags]   (no command: open the app)

COMMANDS
  init        --out DIR [--preset tiny|small]         random byte-level model directory, for from-scratch training
  train       --config run.toml [--resume]            finetune (LoRA or full) from a Hugging Face model directory
  generate    --model DIR --prompt TEXT [--adapter F] [--tokens N] [--temp T] [--top-k K] [--tokenizer PATH|bytes]
  merge       --model DIR --adapter F --out DIR       bake a LoRA adapter into a plain model directory
  parity      --model DIR --ref DIR [--tol 5e-3]      compare logits with PyTorch (see scripts/parity.py)
  demo        [--steps N] [--lora-steps N]            self-contained tour, no downloads (run from the repo root)
  ui          [--port 7860] [--bind IP] [--no-open] [--auto-update]   local web app (run from the folder with your models and data)
  update      [--check] [--repo OWNER/REPO]           install the newest GitHub release over this program
  pull        --repo ORG/NAME [--out DIR]             download a model from Hugging Face (set HF_TOKEN for gated ones)
  worker      --config DIR/cpuforge.toml [--resume]   run a Brewery job (pre-tokenized data, status.json, STOP file)
  version
  bench                                               matmul speed
  bench-train [--layers N] [--seq N] [--rank N]       train-step speed and memory on a SmolLM-135M-shaped model
";

fn main() {
    let result = Args::parse(std::env::args().skip(1)).and_then(|a| match a.command.as_str() {
        "init" => cmd_init::run(&a),
        "train" => cmd_train::run(&a),
        "generate" => cmd_generate::run(&a),
        "merge" => cmd_merge::run(&a),
        "parity" => cmd_parity::run(&a),
        "demo" => cmd_demo::run(&a),
        "ui" => cmd_ui::run(&a),
        "update" => cmd_update::run(&a),
        "pull" => cmd_pull::run(&a),
        "worker" => cmd_worker::run(&a),
        "version" | "--version" => {
            println!("cpuforge {}", cmd_update::version());
            Ok(())
        }
        "bench" => {
            cmd_bench::run_matmul();
            Ok(())
        }
        "bench-train" => cmd_bench::run_train(&a),
        "" => cmd_ui::run(&a), // double-clicking the program opens the app
        "help" | "--help" | "-h" => {
            print!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown command '{other}'\n\n{USAGE}")),
    });
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
