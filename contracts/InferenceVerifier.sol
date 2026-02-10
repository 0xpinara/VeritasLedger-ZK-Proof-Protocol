// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title InferenceVerifier
 * @notice On-chain Groth16 verifier for zkML inference proofs.
 *         Validates that a specific registered model produced a specific output
 *         without revealing model weights or user inputs.
 * @dev    Uses the BN254 pairing precompile (EIP-197) available on Base L2.
 *         Proof consists of three group elements: A ∈ G1, B ∈ G2, C ∈ G1 (192 bytes).
 *         Verification is a single multi-pairing check (~113K gas).
 */
contract InferenceVerifier {

    // ── BN254 curve constants ────────────────────────────────────────────

    uint256 internal constant PRIME_Q =
        21888242871839275222246405745257275088696311157297823662689037894645226208583;

    uint256 internal constant SNARK_SCALAR_FIELD =
        21888242871839275222246405745257275088548364400416034343698204186575808495617;

    // ── Verification key (populated at deployment for the target circuit) ─

    struct VerifyingKey {
        uint256[2]   alpha1;     // [α]₁
        uint256[2][2] beta2;     // [β]₂
        uint256[2][2] gamma2;    // [γ]₂
        uint256[2][2] delta2;    // [δ]₂
        uint256[2][] ic;         // IC points for public inputs
    }

    struct Proof {
        uint256[2]   a;   // A ∈ G1
        uint256[2][2] b;  // B ∈ G2
        uint256[2]   c;   // C ∈ G1
    }

    VerifyingKey internal vk;
    bool public initialized;

    // ── Events ───────────────────────────────────────────────────────────

    event ProofVerified(bytes32 indexed proofHash, bool valid);
    event VerifyingKeySet(uint256 icLength);

    // ── Initialization ───────────────────────────────────────────────────

    /**
     * @notice Set the verifying key for the zkML circuit.
     * @dev    Called once after deployment. IC length = number of public inputs + 1.
     */
    function setVerifyingKey(
        uint256[2] memory _alpha1,
        uint256[2][2] memory _beta2,
        uint256[2][2] memory _gamma2,
        uint256[2][2] memory _delta2,
        uint256[2][] memory _ic
    ) external {
        require(!initialized, "Already initialized");
        vk.alpha1 = _alpha1;
        vk.beta2  = _beta2;
        vk.gamma2 = _gamma2;
        vk.delta2 = _delta2;

        for (uint256 i = 0; i < _ic.length; i++) {
            vk.ic.push(_ic[i]);
        }

        initialized = true;
        emit VerifyingKeySet(_ic.length);
    }

    // ── Core verification ────────────────────────────────────────────────

    /**
     * @notice Verify a Groth16 proof against public inputs.
     * @param proof          The three-element Groth16 proof (A, B, C)
     * @param publicInputs   Array of public inputs [modelWeightHash, outputHash]
     * @return valid          True if the proof is valid
     *
     * @dev Verification equation:
     *      e(A, B) == e(α₁, β₂) · e(Σ aᵢ·ICᵢ, γ₂) · e(C, δ₂)
     *
     *      Rearranged for ecPairing precompile (returns 1 if product of pairings == 1):
     *      e(-A, B) · e(α₁, β₂) · e(vk_x, γ₂) · e(C, δ₂) == 1
     */
    function verifyProof(
        Proof memory proof,
        uint256[] memory publicInputs
    ) public view returns (bool valid) {
        require(initialized, "Verifying key not set");
        require(publicInputs.length + 1 == vk.ic.length, "Input length mismatch");

        // Compute vk_x = IC[0] + Σ publicInputs[i] · IC[i+1]
        uint256[2] memory vk_x = vk.ic[0];
        for (uint256 i = 0; i < publicInputs.length; i++) {
            require(publicInputs[i] < SNARK_SCALAR_FIELD, "Input exceeds field");
            // Scalar multiplication: publicInputs[i] · IC[i+1]
            uint256[2] memory sm = _scalarMul(vk.ic[i + 1], publicInputs[i]);
            // Point addition: vk_x += sm
            vk_x = _pointAdd(vk_x, sm);
        }

        // Negate proof.a for the pairing check
        uint256[2] memory negA = [proof.a[0], PRIME_Q - (proof.a[1] % PRIME_Q)];

        // ecPairing: e(-A, B) · e(α₁, β₂) · e(vk_x, γ₂) · e(C, δ₂) == 1
        uint256[24] memory input;

        // Pair 1: e(-A, B)
        input[0]  = negA[0];
        input[1]  = negA[1];
        input[2]  = proof.b[0][1];  // G2 point: imaginary first
        input[3]  = proof.b[0][0];
        input[4]  = proof.b[1][1];
        input[5]  = proof.b[1][0];

        // Pair 2: e(α₁, β₂)
        input[6]  = vk.alpha1[0];
        input[7]  = vk.alpha1[1];
        input[8]  = vk.beta2[0][1];
        input[9]  = vk.beta2[0][0];
        input[10] = vk.beta2[1][1];
        input[11] = vk.beta2[1][0];

        // Pair 3: e(vk_x, γ₂)
        input[12] = vk_x[0];
        input[13] = vk_x[1];
        input[14] = vk.gamma2[0][1];
        input[15] = vk.gamma2[0][0];
        input[16] = vk.gamma2[1][1];
        input[17] = vk.gamma2[1][0];

        // Pair 4: e(C, δ₂)
        input[18] = proof.c[0];
        input[19] = proof.c[1];
        input[20] = vk.delta2[0][1];
        input[21] = vk.delta2[0][0];
        input[22] = vk.delta2[1][1];
        input[23] = vk.delta2[1][0];

        uint256[1] memory result;
        bool success;

        // Call ecPairing precompile at address 0x08
        assembly {
            success := staticcall(
                gas(),
                0x08,           // ecPairing precompile
                input,
                768,            // 24 * 32 bytes
                result,
                32
            )
        }

        require(success, "Pairing precompile failed");
        return result[0] == 1;
    }

    /**
     * @notice Verify proof and return the proof hash for attestation linking.
     * @param proof          Groth16 proof
     * @param publicInputs   Public inputs [modelWeightHash, outputHash]
     * @return proofHash     keccak256 of the serialized proof
     * @return valid          Verification result
     */
    function verifyAndHash(
        Proof memory proof,
        uint256[] memory publicInputs
    ) external view returns (bytes32 proofHash, bool valid) {
        valid = verifyProof(proof, publicInputs);
        proofHash = keccak256(abi.encode(proof.a, proof.b, proof.c));
        return (proofHash, valid);
    }

    // ── Internal elliptic curve helpers (BN254 precompiles) ──────────────

    function _pointAdd(
        uint256[2] memory p1,
        uint256[2] memory p2
    ) internal view returns (uint256[2] memory r) {
        uint256[4] memory input;
        input[0] = p1[0];
        input[1] = p1[1];
        input[2] = p2[0];
        input[3] = p2[1];

        bool success;
        assembly {
            success := staticcall(gas(), 0x06, input, 128, r, 64)
        }
        require(success, "ecAdd failed");
    }

    function _scalarMul(
        uint256[2] memory p,
        uint256 s
    ) internal view returns (uint256[2] memory r) {
        uint256[3] memory input;
        input[0] = p[0];
        input[1] = p[1];
        input[2] = s;

        bool success;
        assembly {
            success := staticcall(gas(), 0x07, input, 96, r, 64)
        }
        require(success, "ecMul failed");
    }
}
