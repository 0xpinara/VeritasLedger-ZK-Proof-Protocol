// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Votes.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title VRTSToken
 * @notice ERC-20 utility token for the VeritasLedger protocol, deployed on Base L2.
 *         Fixed max supply: 1,000,000,000 VRTS.
 *
 *         Three protocol functions:
 *           1. Staking   – model registrants and provers stake as collateral
 *           2. Fees      – proof generation fees (portion burned, portion to prover)
 *           3. Governance – on-chain voting via ERC20Votes
 *
 * @dev    Includes ERC20Votes for Governor compatibility, ERC20Burnable for the
 *         fee-burn mechanism, and ERC20Permit for gasless approvals.
 *
 *         Burn rate γ (initially 20%) is applied to proof fees.
 *         Staking reward per epoch: R_epoch · (s_i · n_i) / Σ(s_j · n_j)
 */
contract VRTSToken is ERC20, ERC20Burnable, ERC20Votes, ERC20Permit, Ownable {

    // ── Constants ────────────────────────────────────────────────────────

    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 1e18;  // 1 billion tokens
    uint256 public constant MAX_BURN_RATE = 5000;                // 50.00% in bps

    // ── State ────────────────────────────────────────────────────────────

    uint256 public burnRateBps;      // basis points burned per proof fee (initially 2000 = 20%)
    uint256 public totalBurned;
    uint256 public totalMinted;

    // Staking
    mapping(address => uint256) public staked;
    mapping(address => uint256) public proofCount;  // valid proofs per epoch
    uint256 public totalStaked;
    uint256 public epochReward;
    uint256 public currentEpoch;

    // ── Events ───────────────────────────────────────────────────────────

    event FeeBurned(address indexed payer, uint256 burnedAmount, uint256 proverAmount);
    event Staked(address indexed staker, uint256 amount);
    event Unstaked(address indexed staker, uint256 amount);
    event Slashed(address indexed staker, uint256 amount, string reason);
    event BurnRateUpdated(uint256 newBurnRateBps);
    event EpochAdvanced(uint256 newEpoch, uint256 epochReward);

    // ── Constructor ──────────────────────────────────────────────────────

    constructor()
        ERC20("VeritasLedger", "VRTS")
        ERC20Permit("VeritasLedger")
        Ownable(msg.sender)
    {
        burnRateBps = 2000;  // 20%
    }

    // ── Token allocation (owner-controlled minting up to MAX_SUPPLY) ─────

    /**
     * @notice Mint tokens to a recipient. Total minted cannot exceed MAX_SUPPLY.
     * @param to      Recipient address
     * @param amount  Amount to mint (18 decimals)
     */
    function mint(address to, uint256 amount) external onlyOwner {
        require(totalMinted + amount <= MAX_SUPPLY, "Exceeds max supply");
        totalMinted += amount;
        _mint(to, amount);
    }

    // ── Fee processing ───────────────────────────────────────────────────

    /**
     * @notice Process a proof fee. Burns γ% and sends the rest to the prover.
     *         fee(m) = base_fee + α · log₂(constraints(m)) + β · gas_price
     * @param payer   Address paying the fee
     * @param prover  Address of the prover who generated the proof
     * @param amount  Total fee amount in VRTS
     */
    function processProofFee(
        address payer,
        address prover,
        uint256 amount
    ) external {
        uint256 burnAmount   = (amount * burnRateBps) / 10000;
        uint256 proverAmount = amount - burnAmount;

        _burn(payer, burnAmount);
        _transfer(payer, prover, proverAmount);

        totalBurned += burnAmount;

        emit FeeBurned(payer, burnAmount, proverAmount);
    }

    // ── Staking ──────────────────────────────────────────────────────────

    /**
     * @notice Stake VRTS tokens as collateral.
     * @param amount  Amount to stake
     */
    function stake(uint256 amount) external {
        _transfer(msg.sender, address(this), amount);
        staked[msg.sender] += amount;
        totalStaked += amount;
        emit Staked(msg.sender, amount);
    }

    /**
     * @notice Unstake VRTS tokens.
     * @param amount  Amount to unstake
     */
    function unstake(uint256 amount) external {
        require(staked[msg.sender] >= amount, "Insufficient stake");
        staked[msg.sender] -= amount;
        totalStaked -= amount;
        _transfer(address(this), msg.sender, amount);
        emit Unstaked(msg.sender, amount);
    }

    /**
     * @notice Slash a staker's collateral for misbehavior.
     * @param staker    Address to slash
     * @param bps       Slash amount in basis points of their stake
     * @param reason    Human-readable reason for the slash
     */
    function slash(address staker, uint256 bps, string calldata reason) external onlyOwner {
        require(bps <= 10000, "Cannot slash > 100%");
        uint256 slashAmount = (staked[staker] * bps) / 10000;
        staked[staker] -= slashAmount;
        totalStaked -= slashAmount;
        _burn(address(this), slashAmount);
        totalBurned += slashAmount;
        emit Slashed(staker, slashAmount, reason);
    }

    // ── Governance ───────────────────────────────────────────────────────

    /**
     * @notice Update the burn rate (governance-controlled).
     * @param newBurnRateBps  New burn rate in basis points (0–5000)
     */
    function setBurnRate(uint256 newBurnRateBps) external onlyOwner {
        require(newBurnRateBps <= MAX_BURN_RATE, "Exceeds max burn rate");
        burnRateBps = newBurnRateBps;
        emit BurnRateUpdated(newBurnRateBps);
    }

    // ── Required overrides (ERC20 + ERC20Votes) ─────────────────────────

    function _update(
        address from,
        address to,
        uint256 value
    ) internal override(ERC20, ERC20Votes) {
        super._update(from, to, value);
    }

    function nonces(address owner)
        public view override(ERC20Permit, Nonces)
        returns (uint256)
    {
        return super.nonces(owner);
    }
}
