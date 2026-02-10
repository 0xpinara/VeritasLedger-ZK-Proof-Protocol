pragma circom 2.1.6;

include "circomlib/poseidon.circom";

/**
 * Poseidon Hash Configuration for VeritasLedger
 *
 * Parameters (optimized for BN254, per Grassi et al. 2021):
 *   - State width:    t = 3  (2 inputs + 1 capacity) for pairwise hashing
 *                     t = 5  (4 inputs + 1 capacity) for 4-ary Merkle trees
 *   - Full rounds:    R_F = 8  (4 at start, 4 at end)
 *   - Partial rounds: R_P = 57 (128-bit security vs algebraic attacks)
 *   - S-box:          x^5     (smallest power coprime to p - 1)
 *   - R1CS cost:      ~320 constraints per 2-to-1 hash invocation
 *
 * Security guarantees (≥128-bit):
 *   - Gröbner basis attacks
 *   - Interpolation attacks
 *   - Differential / linear cryptanalysis
 *
 * Reference: Whitepaper §18.5
 */

/**
 * PoseidonHash2: Hash two field elements into one.
 * Used for: activation commitments linking adjacent layer proofs.
 *
 * Cost: ~320 R1CS constraints
 */
template PoseidonHash2() {
    signal input  in[2];
    signal output out;

    component hasher = Poseidon(2);
    hasher.inputs[0] <== in[0];
    hasher.inputs[1] <== in[1];
    out <== hasher.out;
}

/**
 * PoseidonHash4: Hash four field elements into one.
 * Used for: 4-ary Merkle tree construction in proof aggregation.
 *
 * Cost: ~480 R1CS constraints (t = 5 configuration)
 */
template PoseidonHash4() {
    signal input  in[4];
    signal output out;

    component hasher = Poseidon(4);
    for (var i = 0; i < 4; i++) {
        hasher.inputs[i] <== in[i];
    }
    out <== hasher.out;
}

/**
 * PoseidonSponge: Variable-length input hashing via sponge construction.
 * Absorbs `n` field elements in pairs, producing a single digest.
 *
 * Used for: hashing model weight matrices (flattened to field element arrays).
 *
 * Cost: ~320 · ceil(n/2) R1CS constraints
 */
template PoseidonSponge(n) {
    signal input  in[n];
    signal output out;

    var nPairs = (n + 1) \ 2;   // ceil(n / 2)

    component hashers[nPairs];
    signal state[nPairs + 1];
    state[0] <== 0;

    for (var i = 0; i < nPairs; i++) {
        hashers[i] = Poseidon(2);

        // First element of pair
        if (2 * i < n) {
            hashers[i].inputs[0] <== in[2 * i] + state[i];
        } else {
            hashers[i].inputs[0] <== state[i];
        }

        // Second element of pair (pad with 0 if odd length)
        if (2 * i + 1 < n) {
            hashers[i].inputs[1] <== in[2 * i + 1];
        } else {
            hashers[i].inputs[1] <== 0;
        }

        state[i + 1] <== hashers[i].out;
    }

    // Finalize: hash the last state with the input length as domain separator
    component finalizer = Poseidon(2);
    finalizer.inputs[0] <== state[nPairs];
    finalizer.inputs[1] <== n;
    out <== finalizer.out;
}

/**
 * ActivationCommitment: Commit to an activation vector for linking
 * adjacent layer proofs in recursive composition (§4.2).
 *
 * h_i = Poseidon(a_i)  is a public output of circuit C_i
 *                       and a public input  of circuit C_{i+1}
 */
template ActivationCommitment(dim) {
    signal input  activation[dim];
    signal output commitment;

    component sponge = PoseidonSponge(dim);
    for (var i = 0; i < dim; i++) {
        sponge.in[i] <== activation[i];
    }
    commitment <== sponge.out;
}
