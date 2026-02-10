"""
token_economics.py — $VRTS Token Economic Model and Simulation

Models the token economics described in Whitepaper §8:
    - Fixed max supply: 1,000,000,000 VRTS
    - Allocation schedule (Table 5)
    - Dynamic proof fee formula (Equation 12)
    - Deflationary burn mechanism (§8.4)
    - Staking reward distribution (Equation 13)
    - Economic security analysis (Proposition 9.4)

Author: Pinar Aksoy
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ── Token parameters (§8.1) ──────────────────────────────────────────────────

MAX_SUPPLY     = 1_000_000_000   # 1 billion VRTS
INITIAL_BURN   = 0.20            # γ = 20% burn rate
MAX_BURN_RATE  = 0.50            # governance ceiling

# Token allocation (Table 5, §8.2)
ALLOCATION = {
    "Community & Ecosystem":  0.30,   # 300M, 5-year linear unlock
    "Protocol Development":   0.20,   # 200M, 4-year linear + 1-year cliff
    "Staking Rewards":        0.20,   # 200M, emitted per epoch
    "Liquidity Provision":    0.10,   # 100M, 50% at launch + 50% over 2 years
    "Team & Advisors":        0.15,   # 150M, 4-year linear + 1-year cliff
    "Reserve":                0.05,   # 50M, DAO-controlled
}


@dataclass
class VestingSchedule:
    """Vesting parameters for each allocation category."""
    total_tokens: float
    cliff_years: float = 0.0
    vesting_years: float = 1.0
    immediate_fraction: float = 0.0  # fraction available at TGE

    def unlocked_at(self, year: float) -> float:
        """Compute tokens unlocked at a given year post-TGE."""
        immediate = self.total_tokens * self.immediate_fraction
        if year < self.cliff_years:
            return immediate
        vested_time = min(year - self.cliff_years, self.vesting_years)
        linear = self.total_tokens * (1 - self.immediate_fraction) * (vested_time / self.vesting_years)
        return immediate + linear


VESTING = {
    "Community & Ecosystem":  VestingSchedule(300e6, cliff_years=0, vesting_years=5),
    "Protocol Development":   VestingSchedule(200e6, cliff_years=1, vesting_years=4),
    "Staking Rewards":        VestingSchedule(200e6, cliff_years=0, vesting_years=10),
    "Liquidity Provision":    VestingSchedule(100e6, cliff_years=0, vesting_years=2,
                                               immediate_fraction=0.5),
    "Team & Advisors":        VestingSchedule(150e6, cliff_years=1, vesting_years=4),
    "Reserve":                VestingSchedule(50e6,  cliff_years=0, vesting_years=10),
}


# ── Proof fee model (§8.3, Equation 12) ──────────────────────────────────────

@dataclass
class FeeParams:
    """Parameters for the dynamic proof generation fee."""
    base_fee: float = 100.0       # VRTS (governance-set, range 10–1000)
    alpha: float = 10.0           # complexity scaling coefficient
    beta: float = 0.5             # gas price coefficient
    gas_price_gwei: float = 0.001 # Base L2 typical gas price


def compute_proof_fee(
    model_constraints: int,
    params: FeeParams = FeeParams()
) -> float:
    """
    Dynamic proof generation fee (Equation 12):

        fee(m) = base_fee + α · log₂(constraints(m)) + β · gas_price

    Parameters:
        model_constraints : R1CS constraint count for the model
        params            : fee parameters

    Returns:
        Fee in VRTS tokens
    """
    log_constraints = math.log2(model_constraints) if model_constraints > 0 else 0
    fee = params.base_fee + params.alpha * log_constraints + params.beta * params.gas_price_gwei
    return fee


# ── Burn mechanism (§8.4) ────────────────────────────────────────────────────

def apply_burn(fee: float, burn_rate: float = INITIAL_BURN) -> Tuple[float, float]:
    """
    Apply deflationary burn to a proof fee.

    Returns:
        (prover_reward, burned_amount)
    """
    burned = fee * burn_rate
    prover_reward = fee - burned
    return prover_reward, burned


# ── Staking reward distribution (§8.5, Equation 13) ──────────────────────────

@dataclass
class Prover:
    """Represents a staked prover in the network."""
    address: str
    stake: float          # VRTS staked
    valid_proofs: int     # proofs generated this epoch
    total_rewards: float = 0.0


def distribute_epoch_rewards(
    provers: List[Prover],
    epoch_reward: float
) -> List[Prover]:
    """
    Distribute staking rewards per Equation 13:

        reward_i = R_epoch · (s_i · n_i) / Σ_j (s_j · n_j)

    where s_i = stake, n_i = valid proofs in epoch.

    Slashing conditions (§8.5):
        - Invalid proof:          10% slash
        - Persistent downtime:     1% slash
        - Unregistered model use: 100% slash
    """
    weighted_sum = sum(p.stake * p.valid_proofs for p in provers)

    if weighted_sum == 0:
        return provers

    for p in provers:
        weight = (p.stake * p.valid_proofs) / weighted_sum
        reward = epoch_reward * weight
        p.total_rewards += reward

    return provers


# ── Economic security analysis (§9.3, Proposition 9.4) ───────────────────────

def attack_cost_lower_bound(
    num_fraudulent_models: int,
    min_stake_vrts: float = 100_000,
    token_price_usd: float = 0.10
) -> float:
    """
    Proposition 9.4: Attack Cost Lower Bound

    Registering n fraudulent models costs ≥ n · S_min · v USD,
    with 100% slashing risk.

    Example: S_min = 100,000 VRTS, v = $0.10, n = 100 → cost ≥ $1M
    """
    return num_fraudulent_models * min_stake_vrts * token_price_usd


# ── Gas cost analysis (Appendix A) ──────────────────────────────────────────

GAS_COSTS = {
    "Model registration (storage write)":  85_000,
    "Groth16 verification (ecPairing)":   113_000,
    "Public input processing":             12_000,
    "Attestation storage":                 65_000,
    "Commit-reveal overhead":              25_000,
}

GAS_COSTS_RECURSIVE = {
    "Recursive proof verification":        95_000,
    "Optimized pairing batch":             85_000,
    "Attestation storage":                 65_000,
}


def compute_verification_cost_usd(
    gas_price_gwei: float = 0.001,
    eth_price_usd: float = 3000.0,
    recursive: bool = True
) -> float:
    """
    Compute on-chain verification cost (Appendix A, Equation 14).

    Non-recursive total: ~280,000 gas
    Recursive total:     ~145,000 gas

    At Base typical gas (0.001 gwei) and ETH $3000:
        145,000 × 0.001 × 10^{-9} × 3000 ≈ $0.000435
    """
    costs = GAS_COSTS_RECURSIVE if recursive else GAS_COSTS
    total_gas = sum(costs.values())
    cost_eth = total_gas * gas_price_gwei * 1e-9
    cost_usd = cost_eth * eth_price_usd
    return cost_usd


# ── Simulation: 10-year token supply projection ─────────────────────────────

def simulate_supply(
    years: int = 10,
    proofs_per_day_initial: int = 1000,
    proof_growth_rate: float = 0.5,        # 50% annual growth
    avg_fee_vrts: float = 150.0,
    burn_rate: float = INITIAL_BURN
) -> Dict[str, List[float]]:
    """
    Simulate circulating supply, burned tokens, and staking dynamics over time.

    Returns time series for:
        - circulating_supply
        - total_burned
        - total_unlocked
        - annual_proof_fees
    """
    results = {
        "year": [],
        "circulating_supply": [],
        "total_burned": [],
        "total_unlocked": [],
        "annual_proof_fees": [],
        "effective_supply": [],
    }

    total_burned = 0.0

    for year in range(years + 1):
        # Compute unlocked tokens across all categories
        total_unlocked = sum(v.unlocked_at(year) for v in VESTING.values())

        # Proof volume and fee revenue
        daily_proofs = proofs_per_day_initial * ((1 + proof_growth_rate) ** year)
        annual_proofs = daily_proofs * 365
        annual_fees = annual_proofs * avg_fee_vrts
        annual_burn = annual_fees * burn_rate

        if year > 0:
            total_burned += annual_burn

        circulating = total_unlocked - total_burned
        effective = max(0, circulating)

        results["year"].append(year)
        results["circulating_supply"].append(total_unlocked)
        results["total_burned"].append(total_burned)
        results["total_unlocked"].append(total_unlocked)
        results["annual_proof_fees"].append(annual_fees)
        results["effective_supply"].append(effective)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("VeritasLedger — $VRTS Token Economics Model")
    print("=" * 72)

    # Token allocation
    print(f"\nToken Allocation (Table 5):")
    print(f"  {'Category':<30} {'Allocation':>12} {'Tokens':>15}")
    print(f"  {'-'*60}")
    for cat, frac in ALLOCATION.items():
        tokens = frac * MAX_SUPPLY
        print(f"  {cat:<30} {frac*100:>11.0f}% {tokens:>15,.0f}")
    print(f"  {'Total':<30} {'100':>11}% {MAX_SUPPLY:>15,.0f}")

    # Proof fee examples
    print(f"\nProof Fee Examples (Equation 12):")
    print(f"  {'Model':<30} {'Constraints':>15} {'Fee (VRTS)':>12}")
    print(f"  {'-'*60}")
    models = [
        ("Linear (1K params)",          4_200),
        ("CNN (100K params)",       8_500_000),
        ("Transformer-small (1M)", 340_000_000),
        ("LLM-7B (recursive)",   2_800_000_000_000),
    ]
    for name, constraints in models:
        fee = compute_proof_fee(constraints)
        print(f"  {name:<30} {constraints:>15,} {fee:>12.1f}")

    # Gas costs
    print(f"\nOn-Chain Verification Costs (Appendix A):")
    for recursive in [False, True]:
        label = "Recursive" if recursive else "Non-recursive"
        costs = GAS_COSTS_RECURSIVE if recursive else GAS_COSTS
        total_gas = sum(costs.values())
        cost_usd = compute_verification_cost_usd(recursive=recursive)
        print(f"  {label}: {total_gas:,} gas → ${cost_usd:.6f}")

    # Economic security
    print(f"\nEconomic Security (Proposition 9.4):")
    for n in [1, 10, 100, 1000]:
        cost = attack_cost_lower_bound(n)
        print(f"  {n:>5} fraudulent models → attack cost ≥ ${cost:>15,.0f}")

    # 10-year supply simulation
    print(f"\nSupply Projection (10-year):")
    sim = simulate_supply()
    print(f"  {'Year':>5} {'Unlocked':>15} {'Burned':>15} {'Effective':>15}")
    print(f"  {'-'*55}")
    for i in range(len(sim["year"])):
        print(f"  {sim['year'][i]:>5} "
              f"{sim['total_unlocked'][i]:>15,.0f} "
              f"{sim['total_burned'][i]:>15,.0f} "
              f"{sim['effective_supply'][i]:>15,.0f}")

    # Staking example
    print(f"\nStaking Reward Distribution (Equation 13):")
    provers = [
        Prover("prover_A", stake=500_000, valid_proofs=100),
        Prover("prover_B", stake=200_000, valid_proofs=250),
        Prover("prover_C", stake=1_000_000, valid_proofs=50),
    ]
    epoch_reward = 50_000  # VRTS per epoch
    provers = distribute_epoch_rewards(provers, epoch_reward)
    print(f"  Epoch reward: {epoch_reward:,} VRTS")
    for p in provers:
        weight = (p.stake * p.valid_proofs) / sum(
            pp.stake * pp.valid_proofs for pp in provers
        )
        print(f"  {p.address}: stake={p.stake:>10,.0f}, proofs={p.valid_proofs:>4}, "
              f"weight={weight:.3f}, reward={p.total_rewards:>10,.1f} VRTS")


if __name__ == "__main__":
    main()
