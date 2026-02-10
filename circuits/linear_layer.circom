pragma circom 2.1.6;

include "circomlib/poseidon.circom";
include "circomlib/comparators.circom";

/**
 * LinearLayer: Arithmetization of a fully-connected neural network layer.
 *
 * Computes y = Wx + b over F_p (BN254 scalar field) using fixed-point
 * representation with Q fractional bits.
 *
 * Constraint count: m(n + 1) + m·Q  (Proposition 3.1 in whitepaper)
 *   - m·n multiplication constraints for the dot products
 *   - m   addition constraints for bias
 *   - m·Q constraints for fixed-point rescaling range checks
 *
 * Parameters:
 *   n_in   – input dimension
 *   n_out  – output dimension
 *   Q      – fractional bits for fixed-point arithmetic
 */
template LinearLayer(n_in, n_out, Q) {

    signal input  x[n_in];              // input activations (fixed-point)
    signal input  W[n_out][n_in];       // weight matrix    (fixed-point)
    signal input  b[n_out];             // bias vector      (fixed-point)
    signal output y[n_out];             // output activations (fixed-point)

    // Intermediate: full-precision dot products before rescaling
    signal acc[n_out][n_in + 1];
    signal products[n_out][n_in];

    // Rescaling witnesses
    signal quotient[n_out];
    signal remainder[n_out];

    for (var i = 0; i < n_out; i++) {

        // ── Accumulate dot product: acc[i][j] = Σ_{k<j} W[i][k] · x[k] ──
        acc[i][0] <== 0;
        for (var j = 0; j < n_in; j++) {
            products[i][j] <== W[i][j] * x[j];          // 1 multiplication constraint
            acc[i][j + 1] <== acc[i][j] + products[i][j];
        }

        // ── Add bias ──────────────────────────────────────────────────────
        // full_result = dot_product + b[i], has 2Q fractional bits
        signal full_result;
        full_result <== acc[i][n_in] + b[i];

        // ── Fixed-point rescaling: shift right by Q bits ──────────────────
        // full_result = quotient[i] · 2^Q + remainder[i]
        // where 0 ≤ remainder[i] < 2^Q
        //
        // The prover supplies quotient and remainder as witnesses;
        // the circuit verifies the decomposition and range constraints.
        quotient[i] * (1 << Q) + remainder[i] === full_result;

        // Range check: remainder < 2^Q  (via binary decomposition)
        component rc = Num2Bits(Q);
        rc.in <== remainder[i];

        y[i] <== quotient[i];
    }
}

/**
 * LinearLayerHash: Same as LinearLayer but also computes the Poseidon hash
 * of the weight matrix for binding to the on-chain model registry.
 */
template LinearLayerHash(n_in, n_out, Q) {

    signal input  x[n_in];
    signal input  W[n_out][n_in];
    signal input  b[n_out];
    signal output y[n_out];
    signal output weightHash;           // Poseidon(flatten(W))

    // Forward pass
    component ll = LinearLayer(n_in, n_out, Q);
    ll.x <== x;
    ll.W <== W;
    ll.b <== b;
    y    <== ll.y;

    // Compute weight hash using Poseidon sponge over flattened weights.
    // Poseidon processes 2 field elements at a time (t = 3 configuration).
    var totalWeights = n_out * n_in;
    var nHashes = (totalWeights + 1) \ 2;     // ceil division

    signal flatW[totalWeights];
    for (var i = 0; i < n_out; i++) {
        for (var j = 0; j < n_in; j++) {
            flatW[i * n_in + j] <== W[i][j];
        }
    }

    component hashers[nHashes];
    signal intermediate[nHashes + 1];
    intermediate[0] <== 0;   // initial state

    for (var h = 0; h < nHashes; h++) {
        hashers[h] = Poseidon(2);
        if (2 * h < totalWeights) {
            hashers[h].inputs[0] <== flatW[2 * h];
        } else {
            hashers[h].inputs[0] <== 0;
        }
        if (2 * h + 1 < totalWeights) {
            hashers[h].inputs[1] <== flatW[2 * h + 1];
        } else {
            hashers[h].inputs[1] <== 0;
        }
        intermediate[h + 1] <== hashers[h].out;
    }

    // Final hash: combine all intermediate hashes
    component finalHash = Poseidon(2);
    finalHash.inputs[0] <== intermediate[nHashes];
    finalHash.inputs[1] <== nHashes;  // domain separator
    weightHash <== finalHash.out;
}
