import random
import math
from typing import List, Tuple

# Probabilities for choosing a Monte Carlo update.
ADD_VERTEX_PROB = 0.125
REMOVE_VERTEX_PROB = 0.25  # Cumulative probability


class Diagram:
    """
    Represents a configuration in the Diagrammatic Monte Carlo simulation
    for a single spin-1/2 particle in a transverse magnetic field.

    The simulation samples diagrams in the path integral expansion of the
    partition function Z = Tr[e^(-beta * H)], where H = h*sigma_z - Gamma*sigma_x.

    Attributes:
        beta (float): Inverse temperature (1/k_B T).
        h (float): Magnetic field strength in the z-direction.
        gamma (float): Transverse field strength (interaction strength) in the x-direction.
        vertices (List[float]): A time-ordered list of interaction vertices (kinks)
                                in the imaginary time interval [0, beta].
        spin (int): The initial spin state at tau=0, either +1 or -1.
        accepted_moves (int): Counter for accepted Monte Carlo moves.
    """

    def __init__(
        self, beta: float, h: float, gamma: float, vertices: List[float] = None
    ):
        """Initializes the Diagram."""
        self.vertices = [] if vertices is None else vertices
        self.beta = beta
        self.h = h
        self.gamma = gamma
        self.spin = 1  # Start with spin up by convention
        self.accepted_moves = 0

    def try_add_vertices(self, warming_up: bool = False):
        """Proposes adding a pair of vertices (kinks) to the worldline."""
        temp_vertices = list(self.vertices)

        # 1. Propose the first vertex tau_a uniformly in [0, beta]
        tau_a = random.uniform(0.0, self.beta)
        temp_vertices.append(tau_a)
        temp_vertices.sort()
        a_index = temp_vertices.index(tau_a)

        # 2. Find the interval [tau_a, tau_f] for the second vertex
        tau_f = (
            self.beta
            if (a_index + 1) == len(temp_vertices)
            else temp_vertices[a_index + 1]
        )

        # 3. Propose the second vertex tau_b in [tau_a, tau_f]
        tau_b = random.uniform(tau_a, tau_f)

        # 4. Calculate acceptance probability
        # The spin in the interval where kinks are added determines the energy change.
        spin_in_interval = self.spin * ((-1.0) ** a_index)

        # CORRECTED: The exponent sign was flipped.
        # The change in the diagonal action is -2 * spin_in_interval * h * (tau_b - tau_a).
        # The weight ratio is exp(-DeltaS), leading to a positive sign in the exponent.
        w_ratio = self.gamma**2 * math.exp(
            2 * self.h * spin_in_interval * (tau_b - tau_a)
        )
        q_ratio = (self.beta * (tau_f - tau_a)) / (len(self.vertices) + 2)

        alpha = min(1.0, w_ratio * q_ratio)

        if alpha > random.random():
            self.vertices.extend([tau_a, tau_b])
            self.vertices.sort()
            if not warming_up:
                self.accepted_moves += 1

    def try_remove_vertices(self, warming_up: bool = False):
        """Proposes removing a pair of adjacent vertices."""
        if len(self.vertices) < 2:
            return

        # 1. Choose a random adjacent pair of vertices to remove
        a_index = random.randrange(len(self.vertices) - 1)
        b_index = a_index + 1

        tau_a = self.vertices[a_index]
        tau_b = self.vertices[b_index]

        # 2. Find the interval length for the proposal ratio
        # The interval tau_f is based on the state *before* removal.
        temp_vertices = list(self.vertices)
        del temp_vertices[b_index]
        del temp_vertices[a_index]

        sort_idx = 0
        while sort_idx < len(temp_vertices) and temp_vertices[sort_idx] < tau_a:
            sort_idx += 1

        tau_f = self.beta if sort_idx == len(temp_vertices) else temp_vertices[sort_idx]

        # 3. Calculate acceptance probability
        spin_in_interval = self.spin * ((-1.0) ** a_index)

        # CORRECTED: The exponent sign was flipped to be the inverse of the add move.
        w_ratio = (1 / self.gamma) ** 2 * math.exp(
            -2 * self.h * spin_in_interval * (tau_b - tau_a)
        )
        q_ratio = (len(self.vertices)) / (self.beta * (tau_f - tau_a))

        alpha = min(1.0, w_ratio * q_ratio)

        if alpha > random.random():
            del self.vertices[b_index]
            del self.vertices[a_index]
            if not warming_up:
                self.accepted_moves += 1

    def _calculate_mz_config(self) -> float:
        """Calculates the magnetization <sigma_z> for the current configuration."""
        sum_alternating_times = sum(
            ((-1.0) ** i) * v for i, v in enumerate(self.vertices)
        )

        # This expression represents (1/beta) * integral_0^beta sigma_z(tau) d(tau)
        integral_val = 2 * sum_alternating_times
        if len(self.vertices) % 2 == 0:
            integral_val += self.beta
        else:  # odd number of vertices, spin is flipped at the end
            integral_val -= self.beta

        return (self.spin / self.beta) * integral_val

    def try_spin_flip(self, warming_up: bool = False):
        """Proposes a global flip of the initial spin s -> -s."""
        # The change in total energy is Delta_E = E_(-s) - E_(s) = -2 * E_(s)
        # E_(s) = h * integral_0^beta sigma_z(tau) d(tau)
        integral_sigma_z = self.beta * self._calculate_mz_config()

        # CORRECTED: The exponent sign was flipped.
        # The acceptance ratio is exp(-Delta_E) = exp(2 * h * integral_sigma_z)
        exponent = 2 * self.h * integral_sigma_z
        w_ratio = math.exp(exponent)

        alpha = min(1.0, w_ratio)
        if alpha > random.random():
            self.spin *= -1
            if not warming_up:
                self.accepted_moves += 1

    def try_update(self, warming_up: bool = False):
        """Randomly chooses and performs one of the three Monte Carlo updates."""
        r = random.random()
        # Always allow adding vertices from an empty configuration
        if len(self.vertices) == 0 or r < ADD_VERTEX_PROB:
            self.try_add_vertices(warming_up)
        elif r < REMOVE_VERTEX_PROB:
            self.try_remove_vertices(warming_up)
        else:
            self.try_spin_flip(warming_up)

    def analytical_solution(self) -> Tuple[float, float]:
        """
        Calculates the analytical solution for the transverse field Ising model magnetization.
        Assumes H = h*sigma_z - gamma*sigma_x.

        Returns:
            A tuple containing (<sigma_x>, <sigma_z>).
        """
        if self.h == 0 and self.gamma == 0:
            return (0.0, 0.0)

        e = math.sqrt(self.h**2 + self.gamma**2)
        # tanh can overflow for large beta*e, handle it by capping at 1.0.
        tanh_val = math.tanh(self.beta * e) if self.beta * e < 20 else 1.0

        # CORRECTED: Signs adjusted for consistency.
        # <sigma_x> is negative, consistent with the H = h*sz - G*sx convention and the MC estimator.
        # <sigma_z> is positive when h > 0.
        mx = (-self.gamma / e) * tanh_val
        mz = (self.h / e) * tanh_val
        return (mx, mz)

    def simulate(self, runs: int, warm_up: int) -> Tuple[float, float, float]:
        """
        Runs the Monte Carlo simulation.

        Args:
            runs (int): The number of Monte Carlo steps for measurement.
            warm_up (int): The number of Monte Carlo steps for thermalization.

        Returns:
            A tuple containing the estimated (<sigma_x>, <sigma_z>) and the acceptance rate.
        """
        # 1. Warm-up phase to reach equilibrium
        for _ in range(warm_up):
            self.try_update(warming_up=True)

        sum_n_vertices = 0
        sum_mz_config = 0

        # 2. Measurement phase
        for _ in range(runs):
            self.try_update()
            sum_n_vertices += len(self.vertices)
            sum_mz_config += self._calculate_mz_config()

        # Estimator for <sigma_x> = -<N_v> / (beta * gamma)
        mx_sim = (
            -(sum_n_vertices / runs) / (self.beta * self.gamma)
            if self.gamma != 0
            else 0
        )

        # Estimator for <sigma_z> = <(1/beta) * integral(sigma_z(tau) d(tau))>
        mz_sim = sum_mz_config / runs

        acceptance_rate = self.accepted_moves / runs

        return (mx_sim, mz_sim, acceptance_rate)


if __name__ == "__main__":
    # --- Simulation Parameters ---
    BETA = 4.0
    H_FIELD = 0.5
    GAMMA_FIELD = 1.0
    SIMULATION_RUNS = 2_000_000
    WARM_UP_STEPS = 100_000

    # --- Analytical Calculation ---
    diagram = Diagram(beta=BETA, h=H_FIELD, gamma=GAMMA_FIELD)
    (mx_anal, mz_anal) = diagram.analytical_solution()

    print("--- Analytical Solution ---")
    print(f"Parameters: beta={BETA}, h={H_FIELD}, gamma={GAMMA_FIELD}")
    print(f"Analytical <sigma_x>: {mx_anal:.6f}")
    print(f"Analytical <sigma_z>: {mz_anal:.6f}")
    print("-" * 27)

    # --- Monte Carlo Simulation ---
    (mx, mz, acceptance_rate) = diagram.simulate(
        runs=SIMULATION_RUNS, warm_up=WARM_UP_STEPS
    )

    print("--- Monte Carlo Simulation ---")
    print(f"Simulation <sigma_x>: {mx:.6f} (Error: {abs(mx - mx_anal):.6f})")
    print(f"Simulation <sigma_z>: {mz:.6f} (Error: {abs(mz - mz_anal):.6f})")
    print(f"Acceptance Rate: {acceptance_rate:.2%}")
    print("-" * 27)
