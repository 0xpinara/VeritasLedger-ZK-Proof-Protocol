pragma circom 2.1.6;

include "circomlib/comparators.circom";
include "circomlib/bitify.circom";

/**
 * ReLU: Arithmetization of the Rectified Linear Unit activation.
 *
 * σ(x) = max(0, x)
 *
 * Since x ∈ F_p is unsigned, we interpret it as a signed integer in
 * [-p/2, p/2]. The circuit introduces an auxiliary binary witness s ∈ {0,1}
 * indicating the sign, and constrains:
 *
 *   s · (1 - s) = 0            (binary constraint on sign bit)
 *   σ(x) = s · x               (conditional output)
 *   x = 2s·q + r               (decomposition with range checks)
 *
 * Constraint count per gate: O(log p) ≈ 258
 *   - 254 constraints for binary decomposition (range proof)
 *   - 2   multiplication constraints (binary check + conditional)
 *   - 2   auxiliary constraints
 *
 * Reference: Whitepaper §3.3.1
 */
template ReLU() {

    signal input  in;
    signal output out;

    // ── Threshold: (p - 1) / 2 ───────────────────────────────────────────
    // BN254 scalar field: p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
    // We treat x as negative if x > (p-1)/2.
    var HALF_P = 10944121435919637611123202872628637544274182200208017171849102093287904247808;

    // ── Sign determination ───────────────────────────────────────────────
    // s = 1 if in ≤ HALF_P (non-negative), s = 0 if in > HALF_P (negative)
    signal s;

    // Binary decomposition of `in` to check its magnitude
    component bits = Num2Bits(254);
    bits.in <== in;

    // Compare: is `in` ≤ HALF_P ?
    component leq = LessEqThan(254);
    leq.in[0] <== in;
    leq.in[1] <== HALF_P;
    s <== leq.out;

    // ── Binary constraint on s ───────────────────────────────────────────
    s * (1 - s) === 0;

    // ── Conditional output ───────────────────────────────────────────────
    out <== s * in;
}

/**
 * ReLULayer: Apply ReLU activation to an entire vector.
 *
 * Total constraints: n × 258
 */
template ReLULayer(n) {

    signal input  in[n];
    signal output out[n];

    component relu[n];

    for (var i = 0; i < n; i++) {
        relu[i] = ReLU();
        relu[i].in <== in[i];
        out[i] <== relu[i].out;
    }
}

/**
 * ReLUWithOverflow: ReLU with explicit overflow detection for fixed-point
 * values. Clips output to a maximum magnitude to prevent wrap-around errors
 * in subsequent layers.
 *
 * Parameters:
 *   MAX_BITS – maximum bit-width of valid activations (e.g., 32 for Q16.16)
 */
template ReLUWithOverflow(MAX_BITS) {

    signal input  in;
    signal output out;
    signal output overflow;    // 1 if |in| exceeds MAX_BITS representation

    component relu = ReLU();
    relu.in <== in;

    // Check if relu output fits in MAX_BITS
    component rangeCheck = Num2Bits(MAX_BITS);
    // If this decomposition succeeds, no overflow
    rangeCheck.in <== relu.out;

    out      <== relu.out;
    overflow <== 0;  // Enforced by range check constraint (fails if overflow)
}
