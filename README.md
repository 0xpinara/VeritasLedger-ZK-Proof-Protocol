# VeritasLedger — Zero-Knowledge Verified AI Inference on Blockchain

**Author:** Pınar Aksoy · pinar0aksoy@gmail.com  
**Status:** Research prototype · Whitepaper v1.0 (December 2025)  
**Chain:** Base (Ethereum L2)

## What This Project Does

VeritasLedger is a protocol that cryptographically proves which AI model produced a given output — without revealing model weights or user inputs. It solves three problems simultaneously:

1. **Model identity verification** — a zk-SNARK proof binds each inference to a hash-locked set of registered weights, so providers cannot swap models.
2. **Content provenance** — every AI-generated output receives an immutable on-chain attestation traceable to its source model.
3. **Privacy by architecture** — queries are encrypted client-side, inference runs inside a zero-knowledge VM, and no user data is collected at any point.

Proof generation is accelerated by a photonic Mach-Zehnder interferometer (MZI) mesh architecture that performs Number-Theoretic Transforms at optical clock rates, achieving 8.6× speedup over GPU-based provers.

## Technical Approach

| Component | Method | Reference |
|-----------|--------|-----------|
| Arithmetization | Neural network layers → R1CS constraints (Groth16/PLONK) | §3 |
| Non-linear activations | Piecewise polynomial lookup tables with range-checked interval selection | §3.3 |
| Fixed-point arithmetic | Q-bit representation over BN254 scalar field, rescaling with range proofs | §3.5, Theorem 3.3 |
| Recursive composition | Per-layer proofs aggregated in a binary tree via IVC; constant-size final proof | §4, Algorithm 1 |
| Photonic NTT | RNS decomposition into 16 channels of 32-bit NTT-friendly primes, MZI butterfly mapping | §5, Appendix C |
| On-chain verification | Groth16 pairing check via EIP-197 precompile on Base L2; ~145K gas recursive | §7, Appendix A |
| Token economics | $VRTS ERC-20: staking, proof fees with 20% burn, epoch-proportional rewards | §8 |

## Repository Structure

```
├── contracts/
│   ├── ModelRegistry.sol          # On-chain model registration with hash-lock and staking
│   ├── InferenceVerifier.sol      # Groth16 proof verification (BN254 pairing precompile)
│   ├── ProvenanceLedger.sol       # Immutable attestation records with commit-reveal
│   └── VRTSToken.sol              # ERC-20 token: staking, fee burn, governance (ERC20Votes)
├── circuits/
│   ├── linear_layer.circom        # R1CS arithmetization of y = Wx + b with fixed-point rescaling
│   ├── relu.circom                # ReLU via sign-bit decomposition (~258 constraints/gate)
│   ├── poseidon_config.circom     # Poseidon hash: t=3, R_F=8, R_P=57, x^5 S-box (~320 constraints)
│   └── recursive_aggregator.circom # Binary-tree proof aggregation with hash-chain linkage
├── src/
│   ├── zkml_compiler.py           # ONNX → R1CS constraint analysis and fixed-point quantization
│   ├── photonic_ntt.py            # Photonic NTT simulation: RNS, MZI model, noise analysis
│   ├── proof_aggregator.py        # Recursive proof composition (Algorithm 1 from whitepaper)
│   └── token_economics.py         # $VRTS supply model, fee formula, staking rewards, gas costs
├── docs/
│   └── VeritasLedger_Whitepaper_v1.0.pdf   # Full whitepaper (38 pages, 25 references)
├── requirements.txt
└── LICENSE
```

## Key Quantitative Results

| Model | Parameters | R1CS Constraints | Prove Time (Photonic) | Proof Size | Verification Gas |
|-------|-----------|-----------------|----------------------|------------|-----------------|
| Linear regressor | 1K | 4.2K | 0.05 s | 192 B | 280K |
| CNN | 100K | 8.5M | 3.5 s | 192 B | 280K |
| Transformer-small | 1M | 340M | 30 s | 192 B | 280K |
| LLaMA-7B (recursive) | 7B | 2.8T | 47 min | 192 B | 145K |

Proof size is constant (192 bytes = 3 BN254 group elements) regardless of model size. Recursive composition reduces verification gas from 280K to 145K via pairing batching.

## Run the Analysis Scripts

```bash
pip install -r requirements.txt

python src/zkml_compiler.py        # Constraint count analysis (reproduces Table 1, Table 6)
python src/photonic_ntt.py         # Photonic NTT benchmarks (reproduces Table 2, Table 3)
python src/token_economics.py      # Token model simulation (reproduces Table 5, Appendix A)
python src/proof_aggregator.py     # Recursive aggregation demo (reproduces Algorithm 1)
```

## Mathematical Foundations

- **Proof system:** Groth16 zk-SNARKs — 3 group elements, single multi-pairing verification, simulation-extractable under AGM + GBM (Theorems 9.1–9.3)
- **Hash function:** Poseidon over BN254 — 128-bit security against Gröbner basis, interpolation, and differential attacks (Grassi et al. 2021)
- **Fixed-point guarantee:** Theorem 3.3 bounds accumulated rounding error to ±1 ULP for B-bit quantized models through L layers with max width d_max, requiring Q = B + ⌈log₂L⌉ + ⌈log₂d_max⌉ fractional bits
- **Privacy:** Theorem 6.2 proves zero-knowledge of user data under Groth16 ZK property + AES-256-GCM semantic security — no PPT adversary gains advantage > negl(λ)

## License

MIT
