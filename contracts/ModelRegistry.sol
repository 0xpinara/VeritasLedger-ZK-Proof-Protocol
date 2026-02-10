// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title ModelRegistry
 * @notice On-chain registry for open-source AI models. Each model is hash-locked
 *         at registration time so that inference proofs can bind to a specific,
 *         immutable set of weights.
 * @dev    Registrants stake VRTS tokens as Sybil resistance and economic commitment.
 */
contract ModelRegistry is Ownable, ReentrancyGuard {

    // ── Types ────────────────────────────────────────────────────────────

    enum ModelState {
        Unregistered,   // 0 – default
        Pending,        // 1 – registered, awaiting confirmation
        Active,         // 2 – confirmed and live
        Disputed,       // 3 – dispute in progress
        Deprecated,     // 4 – publisher-initiated retirement
        Slashed         // 5 – absorbing state (irreversible)
    }

    struct Model {
        bytes32      weightHash;        // Poseidon(W)
        bytes32      architectureHash;  // SHA-256(arch_spec)
        string       ipfsCID;           // IPFS content identifier for weights
        address      registrant;
        uint256      stakeAmount;       // VRTS staked at registration
        uint256      registeredAt;      // block.timestamp
        ModelState   state;
    }

    // ── State ────────────────────────────────────────────────────────────

    uint256 public nextModelId;
    uint256 public minimumStake;                       // in VRTS (18 decimals)
    mapping(uint256 => Model) public models;           // modelId → Model
    mapping(bytes32 => uint256) public hashToModelId;  // weightHash → modelId

    // ── Events ───────────────────────────────────────────────────────────

    event ModelRegistered(
        uint256 indexed modelId,
        bytes32 weightHash,
        bytes32 architectureHash,
        address indexed registrant,
        uint256 stakeAmount
    );
    event ModelConfirmed(uint256 indexed modelId);
    event ModelDisputed(uint256 indexed modelId, address indexed disputer);
    event ModelSlashed(uint256 indexed modelId, uint256 slashedAmount);
    event ModelDeprecated(uint256 indexed modelId);
    event ModelReactivated(uint256 indexed modelId);
    event MinimumStakeUpdated(uint256 newMinimumStake);

    // ── Constructor ──────────────────────────────────────────────────────

    constructor(uint256 _minimumStake) Ownable(msg.sender) {
        minimumStake = _minimumStake;
    }

    // ── External functions ───────────────────────────────────────────────

    /**
     * @notice Register a model by providing its weight hash, architecture hash,
     *         and IPFS CID. Caller must send VRTS stake ≥ minimumStake.
     * @param weightHash       Poseidon hash of the model weights
     * @param archHash         SHA-256 hash of the architecture specification
     * @param ipfsCID          IPFS content identifier where weights are stored
     * @return modelId         Unique identifier assigned to the registered model
     */
    function registerModel(
        bytes32 weightHash,
        bytes32 archHash,
        string calldata ipfsCID
    ) external payable nonReentrant returns (uint256 modelId) {
        require(msg.value >= minimumStake, "Stake below minimum");
        require(hashToModelId[weightHash] == 0, "Weight hash already registered");

        modelId = nextModelId++;

        models[modelId] = Model({
            weightHash:       weightHash,
            architectureHash: archHash,
            ipfsCID:          ipfsCID,
            registrant:       msg.sender,
            stakeAmount:      msg.value,
            registeredAt:     block.timestamp,
            state:            ModelState.Pending
        });

        hashToModelId[weightHash] = modelId;

        emit ModelRegistered(modelId, weightHash, archHash, msg.sender, msg.value);
    }

    /**
     * @notice Confirm a pending model, transitioning it to Active.
     * @param modelId  ID of the model to confirm
     */
    function confirmModel(uint256 modelId) external onlyOwner {
        require(models[modelId].state == ModelState.Pending, "Not pending");
        models[modelId].state = ModelState.Active;
        emit ModelConfirmed(modelId);
    }

    /**
     * @notice Verify that a claimed weight hash matches the registered hash.
     * @param modelId      ID of the registered model
     * @param claimedHash  Hash to compare against registered weightHash
     * @return match        True if hashes match and model is active
     */
    function verifyModelIntegrity(
        uint256 modelId,
        bytes32 claimedHash
    ) external view returns (bool) {
        Model storage m = models[modelId];
        return m.state == ModelState.Active && m.weightHash == claimedHash;
    }

    /**
     * @notice Open a dispute against an active model.
     * @param modelId  ID of the model to dispute
     */
    function disputeModel(uint256 modelId) external {
        require(models[modelId].state == ModelState.Active, "Not active");
        models[modelId].state = ModelState.Disputed;
        emit ModelDisputed(modelId, msg.sender);
    }

    /**
     * @notice Resolve a dispute. If invalid, slash the registrant's stake.
     * @param modelId  ID of the disputed model
     * @param isValid  True → return to Active; False → slash and move to Slashed
     */
    function resolveDispute(uint256 modelId, bool isValid) external onlyOwner {
        require(models[modelId].state == ModelState.Disputed, "Not disputed");

        if (isValid) {
            models[modelId].state = ModelState.Active;
        } else {
            uint256 slashed = models[modelId].stakeAmount;
            models[modelId].stakeAmount = 0;
            models[modelId].state = ModelState.Slashed;
            // Slashed funds go to protocol treasury (this contract balance)
            emit ModelSlashed(modelId, slashed);
        }
    }

    /**
     * @notice Deprecate an active model (publisher only).
     * @param modelId  ID of the model to deprecate
     */
    function deprecateModel(uint256 modelId) external {
        Model storage m = models[modelId];
        require(m.registrant == msg.sender, "Not registrant");
        require(m.state == ModelState.Active, "Not active");
        m.state = ModelState.Deprecated;
        emit ModelDeprecated(modelId);
    }

    /**
     * @notice Reactivate a deprecated model (publisher only).
     * @param modelId  ID of the model to reactivate
     */
    function reactivateModel(uint256 modelId) external {
        Model storage m = models[modelId];
        require(m.registrant == msg.sender, "Not registrant");
        require(m.state == ModelState.Deprecated, "Not deprecated");
        m.state = ModelState.Active;
        emit ModelReactivated(modelId);
    }

    /**
     * @notice Update minimum stake requirement (governance).
     * @param _newMin  New minimum stake in VRTS
     */
    function setMinimumStake(uint256 _newMin) external onlyOwner {
        minimumStake = _newMin;
        emit MinimumStakeUpdated(_newMin);
    }

    // ── View helpers ─────────────────────────────────────────────────────

    function getModel(uint256 modelId) external view returns (Model memory) {
        return models[modelId];
    }

    function isActive(uint256 modelId) external view returns (bool) {
        return models[modelId].state == ModelState.Active;
    }
}
