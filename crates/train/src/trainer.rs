//! The training loop: gradient accumulation, clipping, AdamW with a warmup+cosine
//! schedule, periodic evaluation, and resumable checkpoints.

use std::path::{Path, PathBuf};
use std::time::Instant;

use cf_model::{Grads, MemoryMode, Model, TrainSpec};

use crate::data::Dataset;
use crate::optim::{lr_at, AdamW};
use crate::state::{load_state, save_state};

#[derive(Clone, Debug)]
pub struct TrainConfig {
    /// Optimizer steps to run in total (a resumed run continues up to this number).
    pub steps: usize,
    /// Sequences per optimizer step (gradient accumulation).
    pub batch: usize,
    pub lr: f32,
    pub warmup_steps: usize,
    /// Final learning rate as a fraction of `lr`.
    pub min_lr_ratio: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    /// Clip the global gradient norm to this value; 0 disables.
    pub clip: f32,
    pub memory: MemoryMode,
    pub seed: u64,
    pub log_every: usize,
    /// Evaluate on the validation set every this many steps (0 = never).
    pub eval_every: usize,
    /// Save state every this many steps (0 = only at the end).
    pub save_every: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: 100,
            batch: 4,
            lr: 2e-4,
            warmup_steps: 10,
            min_lr_ratio: 0.1,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            clip: 1.0,
            memory: MemoryMode::Full,
            seed: 1,
            log_every: 1,
            eval_every: 0,
            save_every: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StepStats {
    pub step: usize,
    pub loss: f32,
    pub lr: f32,
    pub grad_norm: f32,
    /// Input tokens processed in this step.
    pub tokens: usize,
    pub tokens_per_sec: f32,
    pub epoch: f32,
}

#[derive(Clone, Debug)]
pub enum Event {
    /// One more sequence of the current step is done (steps can take minutes on big models).
    Micro { step: usize, done: usize, of: usize },
    Step(StepStats),
    Eval { step: usize, val_loss: f32 },
    Saved { step: usize, path: PathBuf },
}

pub struct Trainer {
    pub model: Model,
    pub opt: AdamW,
    pub grads: Grads,
    pub cfg: TrainConfig,
    /// Optimizer steps completed.
    pub step: usize,
    /// Examples drawn from the (shuffled, repeating) training stream so far.
    pub consumed: u64,
    /// Polled between sequences; when it returns true the run saves its state and returns.
    pub should_stop: Option<Box<dyn Fn() -> bool>>,
    /// Set by `train` when it returned early because `should_stop` fired.
    pub stopped: bool,
}

/// Mean loss over up to `max_examples` examples (all of them when 0).
pub fn evaluate(model: &Model, data: &Dataset, max_examples: usize) -> f32 {
    let n = if max_examples == 0 { data.len() } else { data.len().min(max_examples) };
    if n == 0 {
        return 0.0;
    }
    let total: f32 = data.examples[..n].iter().map(|e| model.loss(&e.tokens, &e.targets)).sum();
    total / n as f32
}

impl Trainer {
    pub fn new(model: Model, spec: TrainSpec, cfg: TrainConfig) -> Self {
        let grads = Grads::new(&model, spec);
        let opt = AdamW::new(cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay);
        Trainer { model, opt, grads, cfg, step: 0, consumed: 0, should_stop: None, stopped: false }
    }

    /// Restore progress from a state file written by `save`.
    pub fn resume(&mut self, path: &Path) -> Result<(), String> {
        let (step, consumed) = load_state(path, &mut self.model, &mut self.opt)?;
        self.step = step;
        self.consumed = consumed;
        Ok(())
    }

    pub fn save(&mut self, path: &Path) -> Result<(), String> {
        save_state(path, &mut self.model, &self.grads, &self.opt, self.step, self.consumed)
    }

    /// Run until `cfg.steps` optimizer steps have been taken. If `state_path` is
    /// set, the state is saved there every `save_every` steps and at the end.
    pub fn train(
        &mut self,
        data: &Dataset,
        val: Option<&Dataset>,
        state_path: Option<&Path>,
        on_event: &mut dyn FnMut(&Event),
    ) -> Result<(), String> {
        if data.is_empty() {
            return Err("training set is empty".into());
        }
        let n = data.len() as u64;
        let mut cached_epoch = u64::MAX;
        let mut order: Vec<usize> = Vec::new();

        self.stopped = false;
        'steps: while self.step < self.cfg.steps {
            let started = Instant::now();
            self.grads.zero();
            let (mut loss_sum, mut tokens) = (0.0f32, 0usize);
            let consumed_at_start = self.consumed;
            for done in 0..self.cfg.batch {
                if self.should_stop.as_ref().is_some_and(|f| f()) {
                    // Drop the half-accumulated step so a resumed run repeats it exactly.
                    self.consumed = consumed_at_start;
                    self.stopped = true;
                    if let Some(p) = state_path {
                        self.save(p)?;
                        on_event(&Event::Saved { step: self.step, path: p.to_path_buf() });
                    }
                    break 'steps;
                }
                let (epoch, pos) = (self.consumed / n, (self.consumed % n) as usize);
                if epoch != cached_epoch {
                    order = data.order(self.cfg.seed, epoch);
                    cached_epoch = epoch;
                }
                let ex = &data.examples[order[pos]];
                loss_sum += self.model.forward_backward(&ex.tokens, &ex.targets, &mut self.grads, self.cfg.memory);
                tokens += ex.tokens.len();
                self.consumed += 1;
                on_event(&Event::Micro { step: self.step + 1, done: done + 1, of: self.cfg.batch });
            }
            self.grads.scale(1.0 / self.cfg.batch as f32);
            let grad_norm = self.grads.global_norm();
            if self.cfg.clip > 0.0 && grad_norm > self.cfg.clip {
                self.grads.scale(self.cfg.clip / grad_norm);
            }
            let lr = lr_at(self.step, self.cfg.steps, self.cfg.warmup_steps, self.cfg.lr, self.cfg.min_lr_ratio);
            self.opt.step(&mut self.model, &self.grads, lr);
            self.step += 1;

            if self.cfg.log_every > 0 && (self.step.is_multiple_of(self.cfg.log_every) || self.step == self.cfg.steps) {
                let secs = started.elapsed().as_secs_f32().max(1e-9);
                on_event(&Event::Step(StepStats {
                    step: self.step,
                    loss: loss_sum / self.cfg.batch as f32,
                    lr,
                    grad_norm,
                    tokens,
                    tokens_per_sec: tokens as f32 / secs,
                    epoch: self.consumed as f32 / n as f32,
                }));
            }
            if let Some(v) = val {
                if self.cfg.eval_every > 0 && self.step.is_multiple_of(self.cfg.eval_every) {
                    on_event(&Event::Eval { step: self.step, val_loss: evaluate(&self.model, v, 0) });
                }
            }
            if let Some(p) = state_path {
                let due = self.cfg.save_every > 0 && self.step.is_multiple_of(self.cfg.save_every);
                if due || self.step == self.cfg.steps {
                    self.save(p)?;
                    on_event(&Event::Saved { step: self.step, path: p.to_path_buf() });
                }
            }
        }
        Ok(())
    }
}
