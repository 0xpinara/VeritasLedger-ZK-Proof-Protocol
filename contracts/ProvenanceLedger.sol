// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./ModelRegistry.sol";
import "./InferenceVerifier.sol";

/**
 * @title ProvenanceLedger
 * @notice Immutable ledger recording AI-generated content attestations.
 *         Each attestation links a verified inference proof to a registered model,
 *         creating an on-chain provenance trail.
 * @dev    Implements commit-reveal to prevent front-running and MEV extraction.
 *         No user data, input data, or identity information is stored on-chain.
 */
contract ProvenanceLedger {

    // ── Types ────────────────────────────────────────────────────────────

    struct Attestation {
        bytes32   attestationId;
        uint256   modelId;
        bytes32   outputHash;     // Poseidon(y)
        bytes32   proofHash;      // keccak256(π)
        uint256   blockNumber;
        uint256   timestamp;
        // NOTE: No user data, no input data
    }

    struct Commitment {
        bytes32   commitHash;     // Poseidon(π ‖ nonce)
        uint256   blockNumber;    // block at which commitment was made
        bool      revealed;
    }

    // ── State ────────────────────────────────────────────────────────────

    ModelRegistry public modelRegistry;
    InferenceVerifier public verifier;

    uint256 public attestationCount;
    mapping(bytes32 => Attestation) public attestations;        // attestationId → Attestation
    mapping(uint256 => bytes32[])   public modelAttestations;   // modelId → attestationIds
    mapping(address => Commitment)  public commitments;         // prover → pending commitment

    uint256 public constant COMMIT_REVEAL_WINDOW = 256;         // blocks

    // ── Events ───────────────────────────────────────────────────────────

    event CommitmentSubmitted(address indexed prover, bytes32 commitHash);
    event AttestationRecorded(
        bytes32 indexed attestationId,
        uint256 indexed modelId,
        bytes32 outputHash,
        bytes32 proofHash,
        uint256 blockNumber,
        uint256 timestamp
    );

    // ── Constructor ──────────────────────────────────────────────────────

    constructor(address _modelRegistry, address _verifier) {
        modelRegistry = ModelRegistry(_modelRegistry);
        verifier      = InferenceVerifier(_verifier);
    }

    // ── Commit phase ─────────────────────────────────────────────────────

    /**
     * @notice Submit a commitment hash before revealing the proof.
     *         commitHash = keccak256(abi.encodePacked(proofA, proofB, proofC, nonce))
     * @param commitHash  Hash binding the prover to a specific proof
     */
    function submitCommitment(bytes32 commitHash) external {
        require(commitments[msg.sender].commitHash == bytes32(0), "Existing commitment");
        commitments[msg.sender] = Commitment({
            commitHash:  commitHash,
            blockNumber: block.number,
            revealed:    false
        });
        emit CommitmentSubmitted(msg.sender, commitHash);
    }

    // ── Reveal phase + attestation ───────────────────────────────────────

    /**
     * @notice Reveal the proof, verify it on-chain, and record the attestation.
     * @param modelId       ID of the registered model
     * @param outputHash    Poseidon hash of the inference output
     * @param proof         Groth16 proof (A, B, C)
     * @param publicInputs  [modelWeightHash, outputHash] as field elements
     * @param nonce         Random nonce used in the commitment
     * @return attestationId  Unique identifier for the recorded attestation
     */
    function revealAndAttest(
        uint256 modelId,
        bytes32 outputHash,
        InferenceVerifier.Proof memory proof,
        uint256[] memory publicInputs,
        bytes32 nonce
    ) external returns (bytes32 attestationId) {
        // 1. Verify commitment exists and is within the reveal window
        Commitment storage c = commitments[msg.sender];
        require(c.commitHash != bytes32(0), "No commitment");
        require(!c.revealed, "Already revealed");
        require(block.number > c.blockNumber, "Same block");
        require(block.number <= c.blockNumber + COMMIT_REVEAL_WINDOW, "Window expired");

        // 2. Verify commitment matches revealed data
        bytes32 expectedCommit = keccak256(
            abi.encodePacked(proof.a, proof.b, proof.c, nonce)
        );
        require(expectedCommit == c.commitHash, "Commitment mismatch");

        // 3. Verify model is active
        require(modelRegistry.isActive(modelId), "Model not active");

        // 4. Verify the Groth16 proof on-chain
        (bytes32 proofHash, bool valid) = verifier.verifyAndHash(proof, publicInputs);
        require(valid, "Proof invalid");

        // 5. Record attestation
        attestationId = keccak256(
            abi.encodePacked(modelId, outputHash, proofHash, block.number)
        );

        attestations[attestationId] = Attestation({
            attestationId: attestationId,
            modelId:       modelId,
            outputHash:    outputHash,
            proofHash:     proofHash,
            blockNumber:   block.number,
            timestamp:     block.timestamp
        });

        modelAttestations[modelId].push(attestationId);
        attestationCount++;

        // 6. Mark commitment as revealed
        c.revealed = true;

        emit AttestationRecorded(
            attestationId, modelId, outputHash, proofHash, block.number, block.timestamp
        );
    }

    // ── Query functions ──────────────────────────────────────────────────

    /**
     * @notice Retrieve an attestation by its ID.
     * @param attestationId  The attestation identifier
     * @return att            Full attestation struct
     */
    function getAttestation(bytes32 attestationId)
        external view returns (Attestation memory att)
    {
        att = attestations[attestationId];
        require(att.attestationId != bytes32(0), "Attestation not found");
    }

    /**
     * @notice Verify that candidate output matches an attestation's recorded hash.
     * @param attestationId   The attestation to check against
     * @param candidateHash   Poseidon(candidate_output)
     * @return matches         True if hashes match
     */
    function verifyOutput(bytes32 attestationId, bytes32 candidateHash)
        external view returns (bool)
    {
        return attestations[attestationId].outputHash == candidateHash;
    }

    /**
     * @notice Get all attestation IDs for a given model.
     * @param modelId  The registered model ID
     * @return ids      Array of attestation identifiers
     */
    function getModelAttestations(uint256 modelId)
        external view returns (bytes32[] memory)
    {
        return modelAttestations[modelId];
    }
}
