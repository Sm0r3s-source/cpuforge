//! Causal grouped-query attention, forward and backward.
//!
//! Layouts (all row-major, one sequence):
//!   q, o, dq, do : [seq, n_heads, head_dim]
//!   k, v, dk, dv : [seq, n_kv,    head_dim]
//!   lse          : [n_heads, seq]            (log-sum-exp of each score row)
//!
//! Flash-style memory: only `lse` is kept between forward and backward; the
//! probabilities are recomputed in backward. Work is done per head and per
//! block of queries, so the transient score matrix is `q_block x seq` and
//! blocks only touch the keys they can see (about 40% fewer FLOPs than a full
//! square at seq=512). Heads are views into the packed tensors via strides, so
//! nothing is copied.

use crate::matmul::{gemm_strided, Mat};

/// Query rows processed together. Bigger = better gemm efficiency, more scratch.
pub const DEFAULT_Q_BLOCK: usize = 128;

#[derive(Clone, Copy, Debug)]
pub struct AttnShape {
    pub seq: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
}

impl AttnShape {
    fn q_stride(&self) -> usize {
        self.n_heads * self.head_dim
    }
    fn kv_stride(&self) -> usize {
        self.n_kv * self.head_dim
    }
    fn check(&self, q: usize, k: usize) {
        assert!(self.n_kv > 0 && self.n_heads.is_multiple_of(self.n_kv), "n_heads must be a multiple of n_kv");
        assert_eq!(q, self.seq * self.q_stride());
        assert_eq!(k, self.seq * self.kv_stride());
    }
}

pub fn attention_forward(q: &[f32], k: &[f32], v: &[f32], o: &mut [f32], lse: &mut [f32], s: AttnShape) {
    attention_forward_blocked(q, k, v, o, lse, s, DEFAULT_Q_BLOCK)
}

pub fn attention_forward_blocked(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    o: &mut [f32],
    lse: &mut [f32],
    s: AttnShape,
    q_block: usize,
) {
    s.check(q.len(), k.len());
    assert_eq!(v.len(), k.len());
    assert_eq!(o.len(), q.len());
    assert_eq!(lse.len(), s.n_heads * s.seq);
    let (t_len, hd) = (s.seq, s.head_dim);
    let (qs, kvs) = (s.q_stride() as isize, s.kv_stride() as isize);
    let group = s.n_heads / s.n_kv;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut p = vec![0.0f32; q_block.min(t_len.max(1)) * t_len];

    for h in 0..s.n_heads {
        let g = h / group;
        let mut t0 = 0;
        while t0 < t_len {
            let t1 = (t0 + q_block).min(t_len);
            let (bq, l) = (t1 - t0, t1);

            // S = Q_blk · K[0..l]ᵀ
            gemm_strided(
                Mat::new(&q[t0 * qs as usize + h * hd..], qs, 1),
                Mat::new(&k[g * hd..], 1, kvs),
                &mut p[..bq * l],
                l as isize,
                1,
                bq,
                hd,
                l,
                false,
            );

            // scale + causal mask + softmax, row by row
            for r in 0..bq {
                let valid = t0 + r + 1;
                let row = &mut p[r * l..(r + 1) * l];
                let mut max = f32::NEG_INFINITY;
                for x in row[..valid].iter_mut() {
                    *x *= scale;
                    max = max.max(*x);
                }
                let mut sum = 0.0f32;
                for x in row[..valid].iter_mut() {
                    *x = (*x - max).exp();
                    sum += *x;
                }
                lse[h * t_len + t0 + r] = max + sum.ln();
                let inv = 1.0 / sum;
                for x in row[..valid].iter_mut() {
                    *x *= inv;
                }
                row[valid..].fill(0.0);
            }

            // O_blk = P · V[0..l]
            gemm_strided(
                Mat::new(&p[..bq * l], l as isize, 1),
                Mat::new(&v[g * hd..], kvs, 1),
                &mut o[t0 * qs as usize + h * hd..],
                qs,
                1,
                bq,
                l,
                hd,
                false,
            );
            t0 = t1;
        }
    }
}

/// Accumulates into `dq`, `dk`, `dv`.
#[allow(clippy::too_many_arguments)]
pub fn attention_backward(
    d_o: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    o: &[f32],
    lse: &[f32],
    dq: &mut [f32],
    dk: &mut [f32],
    dv: &mut [f32],
    s: AttnShape,
) {
    attention_backward_blocked(d_o, q, k, v, o, lse, dq, dk, dv, s, DEFAULT_Q_BLOCK)
}

#[allow(clippy::too_many_arguments)]
pub fn attention_backward_blocked(
    d_o: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    o: &[f32],
    lse: &[f32],
    dq: &mut [f32],
    dk: &mut [f32],
    dv: &mut [f32],
    s: AttnShape,
    q_block: usize,
) {
    s.check(q.len(), k.len());
    assert!(v.len() == k.len() && dk.len() == k.len() && dv.len() == k.len());
    assert!(d_o.len() == q.len() && o.len() == q.len() && dq.len() == q.len());
    assert_eq!(lse.len(), s.n_heads * s.seq);
    let (t_len, hd) = (s.seq, s.head_dim);
    let (qs, kvs) = (s.q_stride() as isize, s.kv_stride() as isize);
    let group = s.n_heads / s.n_kv;
    let scale = 1.0 / (hd as f32).sqrt();
    let cap = q_block.min(t_len.max(1)) * t_len;
    let mut p = vec![0.0f32; cap];
    let mut ds = vec![0.0f32; cap];

    for h in 0..s.n_heads {
        let g = h / group;
        let mut t0 = 0;
        while t0 < t_len {
            let t1 = (t0 + q_block).min(t_len);
            let (bq, l) = (t1 - t0, t1);
            let q_off = t0 * qs as usize + h * hd;

            // Recompute P from the saved log-sum-exp.
            gemm_strided(
                Mat::new(&q[q_off..], qs, 1),
                Mat::new(&k[g * hd..], 1, kvs),
                &mut p[..bq * l],
                l as isize,
                1,
                bq,
                hd,
                l,
                false,
            );
            for r in 0..bq {
                let valid = t0 + r + 1;
                let row = &mut p[r * l..(r + 1) * l];
                let lse_r = lse[h * t_len + t0 + r];
                for x in row[..valid].iter_mut() {
                    *x = (*x * scale - lse_r).exp();
                }
                row[valid..].fill(0.0);
            }

            // dV[0..l] += Pᵀ · dO_blk   (A = P viewed as [l, bq])
            gemm_strided(
                Mat::new(&p[..bq * l], 1, l as isize),
                Mat::new(&d_o[q_off..], qs, 1),
                &mut dv[g * hd..],
                kvs,
                1,
                l,
                bq,
                hd,
                true,
            );

            // dP = dO_blk · V[0..l]ᵀ
            gemm_strided(
                Mat::new(&d_o[q_off..], qs, 1),
                Mat::new(&v[g * hd..], 1, kvs),
                &mut ds[..bq * l],
                l as isize,
                1,
                bq,
                hd,
                l,
                false,
            );

            // dS = P ⊙ (dP − D) · scale,  with D = rowsum(dO ⊙ O)
            for r in 0..bq {
                let off = q_off + r * qs as usize;
                let dsum: f32 = d_o[off..off + hd].iter().zip(&o[off..off + hd]).map(|(a, b)| a * b).sum();
                let prow = &p[r * l..(r + 1) * l];
                let drow = &mut ds[r * l..(r + 1) * l];
                for j in 0..l {
                    drow[j] = prow[j] * (drow[j] - dsum) * scale;
                }
            }

            // dQ_blk += dS · K[0..l]
            gemm_strided(
                Mat::new(&ds[..bq * l], l as isize, 1),
                Mat::new(&k[g * hd..], kvs, 1),
                &mut dq[q_off..],
                qs,
                1,
                bq,
                l,
                hd,
                true,
            );

            // dK[0..l] += dSᵀ · Q_blk
            gemm_strided(
                Mat::new(&ds[..bq * l], 1, l as isize),
                Mat::new(&q[q_off..], qs, 1),
                &mut dk[g * hd..],
                kvs,
                1,
                l,
                bq,
                hd,
                true,
            );
            t0 = t1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::*;
    use crate::rng::Rng;

    /// Naive f64 attention with explicit GQA head mapping. Returns O.
    fn attn_ref(q: &[f64], k: &[f64], v: &[f64], s: AttnShape) -> Vec<f64> {
        let (t_len, hd) = (s.seq, s.head_dim);
        let group = s.n_heads / s.n_kv;
        let mut o = vec![0.0; t_len * s.n_heads * hd];
        for h in 0..s.n_heads {
            let g = h / group;
            for t in 0..t_len {
                let qv = &q[(t * s.n_heads + h) * hd..][..hd];
                let mut scores = Vec::new();
                for j in 0..=t {
                    let kv = &k[(j * s.n_kv + g) * hd..][..hd];
                    scores.push(qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f64>() / (hd as f64).sqrt());
                }
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = scores.iter().map(|x| (x - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                for (j, e) in exps.iter().enumerate() {
                    let vv = &v[(j * s.n_kv + g) * hd..][..hd];
                    for d in 0..hd {
                        o[(t * s.n_heads + h) * hd + d] += e / sum * vv[d];
                    }
                }
            }
        }
        o
    }

    fn rand(n: usize, seed: u64, std: f32) -> Vec<f32> {
        let mut r = Rng::new(seed);
        (0..n).map(|_| r.normal() * std).collect()
    }

    struct Case {
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        s: AttnShape,
    }

    fn case(seq: usize, n_heads: usize, n_kv: usize, hd: usize, seed: u64) -> Case {
        let s = AttnShape { seq, n_heads, n_kv, head_dim: hd };
        Case {
            q: rand(seq * n_heads * hd, seed, 1.0),
            k: rand(seq * n_kv * hd, seed + 1, 1.0),
            v: rand(seq * n_kv * hd, seed + 2, 1.0),
            s,
        }
    }

    // (seq, heads, kv heads, head_dim, q_block): multiple blocks, ragged last block, GQA, MHA, MQA.
    const CASES: &[(usize, usize, usize, usize, usize)] =
        &[(8, 4, 2, 6, 3), (10, 3, 1, 4, 4), (7, 2, 2, 8, 100), (13, 6, 2, 4, 5), (1, 2, 1, 4, 4)];

    #[test]
    fn forward_matches_f64_reference() {
        for (i, &(seq, h, kv, hd, qb)) in CASES.iter().enumerate() {
            let c = case(seq, h, kv, hd, 10 + i as u64);
            let mut o = vec![0.0; seq * h * hd];
            let mut lse = vec![0.0; h * seq];
            attention_forward_blocked(&c.q, &c.k, &c.v, &mut o, &mut lse, c.s, qb);
            let want = attn_ref(&to_f64(&c.q), &to_f64(&c.k), &to_f64(&c.v), c.s);
            assert_close(&format!("attn fwd case {i}"), &o, &want, 1e-4);
        }
    }

    #[test]
    fn block_size_does_not_change_result() {
        let c = case(37, 4, 2, 8, 30);
        let run = |qb: usize| {
            let mut o = vec![0.0; 37 * 4 * 8];
            let mut lse = vec![0.0; 4 * 37];
            attention_forward_blocked(&c.q, &c.k, &c.v, &mut o, &mut lse, c.s, qb);
            (o, lse)
        };
        let (o1, l1) = run(1);
        let (o2, l2) = run(16);
        let (o3, l3) = run(1000);
        for i in 0..o1.len() {
            assert!((o1[i] - o2[i]).abs() < 1e-5 && (o1[i] - o3[i]).abs() < 1e-5);
        }
        for i in 0..l1.len() {
            assert!((l1[i] - l2[i]).abs() < 1e-5 && (l1[i] - l3[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn first_token_attends_only_to_itself() {
        let c = case(5, 2, 1, 4, 40);
        let mut o = vec![0.0; 5 * 2 * 4];
        let mut lse = vec![0.0; 2 * 5];
        attention_forward(&c.q, &c.k, &c.v, &mut o, &mut lse, c.s);
        // t=0: output equals v[0] of the shared kv head, for both q heads.
        for h in 0..2 {
            for d in 0..4 {
                assert!((o[h * 4 + d] - c.v[d]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn backward_matches_finite_differences() {
        for (i, &(seq, h, kv, hd, qb)) in CASES.iter().enumerate() {
            let c = case(seq, h, kv, hd, 50 + i as u64);
            let g = rand(seq * h * hd, 70 + i as u64, 1.0);

            let mut o = vec![0.0; seq * h * hd];
            let mut lse = vec![0.0; h * seq];
            attention_forward_blocked(&c.q, &c.k, &c.v, &mut o, &mut lse, c.s, qb);
            let mut dq = vec![0.0; c.q.len()];
            let mut dk = vec![0.0; c.k.len()];
            let mut dv = vec![0.0; c.v.len()];
            attention_backward_blocked(&g, &c.q, &c.k, &c.v, &o, &lse, &mut dq, &mut dk, &mut dv, c.s, qb);

            let (q64, k64, v64, g64) = (to_f64(&c.q), to_f64(&c.k), to_f64(&c.v), to_f64(&g));
            let loss = |o: &[f64]| o.iter().zip(&g64).map(|(a, b)| a * b).sum::<f64>();
            let nq = numeric_grad(|x| loss(&attn_ref(x, &k64, &v64, c.s)), &q64, 1e-6);
            let nk = numeric_grad(|x| loss(&attn_ref(&q64, x, &v64, c.s)), &k64, 1e-6);
            let nv = numeric_grad(|x| loss(&attn_ref(&q64, &k64, x, c.s)), &v64, 1e-6);
            assert_grad_close(&format!("attn dq case {i}"), &dq, &nq, 1e-3);
            assert_grad_close(&format!("attn dk case {i}"), &dk, &nk, 1e-3);
            assert_grad_close(&format!("attn dv case {i}"), &dv, &nv, 1e-3);
        }
    }
}
