import random
import math


class Diagram:
    def __init__(self, beta, h, gamma, vertices=None):
        self.vertices = [] if vertices is None else vertices
        self.beta = beta
        self.h = h
        self.gamma = gamma
        self.spin = -1
        self.accepted_moves = 0

    def try_add_vertices(self, warming_up=False):
        """Proposes adding a pair of vertices (kinks) to the worldline."""
        temp_vertices = list(self.vertices)  # Copies the list

        # 1. Propose the first vertex tau_a uniformly in [0, beta]
        tau_a = random.uniform(0.0, self.beta)
        temp_vertices.append(tau_a)
        temp_vertices.sort()

        # 2. Find the interval [tau_a, tau_f] for the second vertex
        a_index = temp_vertices.index(tau_a)
        tau_f = (
            self.beta
            if (a_index + 1) == len(temp_vertices)
            else temp_vertices[a_index + 1]
        )

        # 3. Propose the second vertex tau_b in [tau_a, tau_f]
        tau_b = random.uniform(tau_a, tau_f)
        # Weight ratio from the Hamiltonian
        w_ratio = self.gamma**2 * math.exp(
            -2 * self.h * self.spin * ((-1.0) ** a_index) * (tau_b - tau_a)
        )
        q_ratio = (self.beta * (tau_f - tau_a)) / (len(self.vertices) + 2)

        alpha = min(1.0, w_ratio * q_ratio)

        if alpha > random.random():
            self.vertices.append(tau_a)
            self.vertices.append(tau_b)
            self.vertices.sort()
            if not warming_up:
                self.accepted_moves += 1

    def try_remove_vertices(self, warming_up=False):
        """Proposes removing a pair of adjacent vertices."""
        # Can't remove if there are fewer than 2 vertices
        if len(self.vertices) < 2:
            return

        # 1. Choose a random adjacent pair of vertices to remove
        # There are len(vertices)-1 such pairs.
        a_index = random.randrange(len(self.vertices) - 1)
        b_index = a_index + 1

        tau_a = self.vertices[a_index]
        tau_b = self.vertices[b_index]

        # 2. Determine the next vertex to compute the proposal interval length
        tau_f = (
            self.vertices[b_index + 1]
            if (b_index + 1 < len(self.vertices))
            else self.beta
        )

        # 3. Calculate acceptance probability
        # Parity determines the spin state in the interval [tau_a, tau_b]
        parity = a_index
        w_ratio = (1 / self.gamma) ** 2 * math.exp(
            2 * self.h * self.spin * ((-1.0) ** parity) * (tau_b - tau_a)
        )

        q_ratio = len(self.vertices) / (self.beta * (tau_f - tau_a))

        alpha = min(1.0, w_ratio * q_ratio)

        if alpha > random.random():
            # Remove the two vertices
            del self.vertices[b_index]
            del self.vertices[a_index]
            if not warming_up:
                self.accepted_moves += 1

    def try_spin_flip(self, warming_up=False):
        sum_alternating_times = 0
        for i, vertex_time in enumerate(self.vertices):
            sum_alternating_times += ((-1.0) ** i) * vertex_time

        exponent = -2 * self.h * self.spin * (2 * sum_alternating_times + self.beta)
        w_ratio = math.exp(exponent)

        alpha = min(1.0, w_ratio)
        if alpha > random.random():
            self.spin *= -1
            if not warming_up:
                self.accepted_moves += 1

    def try_update(self, warming_up=False):
        """Randomly choose one of the three update types."""
        r = random.random()

        if len(self.vertices) == 0 or r < 0.45:
            self.try_add_vertices(warming_up)
        elif r < 0.9:
            self.try_remove_vertices(warming_up)
        else:
            self.try_spin_flip(warming_up)

    def analytical_solution(self):
        """
        Analytical solution for the transverse field Ising model magnetization.
        Returns a tuple containing (<sigma_x>, <sigma_z>).
        """
        e = math.sqrt(self.h**2 + self.gamma**2)
        # tanh can overflow for large beta*e, handle it.
        tanh_val = math.tanh(self.beta * e) if self.beta * e < 100 else 1.0

        mx = (-self.gamma / e) * tanh_val
        mz = (-self.h / e) * tanh_val
        return (mx, mz)

    def simulate(self, runs, warm_up):
        """Run the simulation for a number of steps."""
        for i in range(warm_up):
            self.try_update(warming_up=True)

        sum_n_vertices = 0
        sum_mz_config = 0

        for _ in range(runs):
            self.try_update()

            # Accumulate metrics for <sigma_x>
            sum_n_vertices += len(self.vertices)

            # Accumulate metrics for <sigma_z>
            sum_alternating_times = 0
            for j, vertex_time in enumerate(self.vertices):
                sum_alternating_times += ((-1.0) ** j) * vertex_time

            mz_config = (self.spin / self.beta) * (
                2 * sum_alternating_times + self.beta
            )
            sum_mz_config += mz_config

        return (
            -(sum_n_vertices / runs) / (self.beta * self.gamma),
            sum_mz_config / runs,
            self.accepted_moves / runs,
        )


if __name__ == "__main__":
    # Diagram init
    diagram = Diagram(beta=0.5, h=1, gamma=0.4)
    (mx_anal, mz_anal) = diagram.analytical_solution()

    print("--- Analytical Solution ---")
    print(f"Analytical value for <sigma_x>: {mx_anal:.6f}")
    print(f"Analytical value for <sigma_z>: {mz_anal:.6f}")
    print("-" * 27)

    # Simulation
    (mx, mz, acceptance_rate) = diagram.simulate(runs=1_000_000, warm_up=50_000)

    print("--- Monte Carlo Simulation ---")
    print(f"Simulation value for <sigma_x>: {mx:.6f}")
    print(f"Simulation value for <sigma_z>: {mz:.6f}")
    print(f"Acceptance Rate: {acceptance_rate:.2%}")
    print("-" * 27)
