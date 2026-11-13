"""
PaiGen AI: Mathematical Foundations & Theoretical Physics Epistemology
Architect: Arjun Singh

Encodes pure mathematics, engineering mathematics, and theoretical physics
from Ramanujan's modular forms to Hawking's singularity theorems, equipping
PaiGen's brain with deep analytical rigor and mathematical pattern intuition.
"""

from typing import Dict, List
from pydantic import BaseModel, Field


class MathematicalDomain(BaseModel):
    domain_id: str
    name: str
    category: str  # "Pure Mathematics", "Engineering Mathematics", "Theoretical Physics & Cosmology"
    titans: List[str]  # e.g., ["Srinivasa Ramanujan", "Carl Friedrich Gauss"]
    core_theorems_and_equations: List[str]
    intuitive_breakthrough: str
    reasoning_power_for_ai: str
    tags: List[str] = Field(default_factory=list)


MATHEMATICS_KNOWLEDGE_BASE: List[MathematicalDomain] = [
    # =========================================================================
    # 1. NUMBER THEORY & MODULAR FORMS (RAMANUJAN & GAUSS)
    # =========================================================================
    MathematicalDomain(
        domain_id="math_number_theory_ramanujan",
        name="Analytic Number Theory & Modular Forms",
        category="Pure Mathematics",
        titans=["Srinivasa Ramanujan", "Carl Friedrich Gauss", "Bernhard Riemann", "Pierre de Fermat"],
        core_theorems_and_equations=[
            "Ramanujan's Partition Asymptotics: p(n) ~ (1 / (4n * sqrt(3))) * exp(pi * sqrt(2n/3))",
            "Ramanujan Mock Theta Functions and Continued Fractions",
            "Riemann Zeta Function & Hypothesis: zeta(s) = sum(1/n^s) = prod(1 / (1 - p^(-s)))",
            "Euler-Riemann Prime Distribution Connection: pi(x) ~ Li(x)"
        ],
        intuitive_breakthrough=(
            "Numbers are not discrete isolated islands; they have a deep, continuous, harmonic architecture. "
            "Modular forms possess profound symmetries that bridge seemingly unrelated fields (like geometry and prime numbers)."
        ),
        reasoning_power_for_ai=(
            "Inspires intuitive, non-linear pattern recognition. Teaches PaiGen to identify deep mathematical "
            "isomorphisms and harmonic symmetries across disparate sensory data streams."
        ),
        tags=["number_theory", "ramanujan", "primes", "modular_forms", "pure_math"]
    ),

    # =========================================================================
    # 2. ABSTRACT ALGEBRA & SYMMETRY (NOETHER & GALOIS)
    # =========================================================================
    MathematicalDomain(
        domain_id="math_abstract_algebra_noether",
        name="Abstract Algebra, Group Theory & Symmetries",
        category="Pure Mathematics",
        titans=["Emmy Noether", "Évariste Galois", "Niels Henrik Abel", "Sophus Lie"],
        core_theorems_and_equations=[
            "Noether's First Theorem: Every continuous differentiable symmetry of action yields a corresponding conservation law (Time translation -> Energy; Spatial translation -> Momentum)",
            "Galois Theory: Solvability of polynomial equations by radicals via automorphism groups (unsolvability of quintics)",
            "Lie Groups & Algebras (SU(2), SU(3), SO(3)): Continuous transformations governing quantum spin and particle physics"
        ],
        intuitive_breakthrough=(
            "Structure and symmetry are the fundamental grammar of reality. What remains invariant under transformation "
            "reveals the deepest conservation laws of nature."
        ),
        reasoning_power_for_ai=(
            "Enables PaiGen to construct invariant mental representations: recognizing consistent user identity, "
            "emotional equilibrium, and systemic invariants regardless of surface noise."
        ),
        tags=["algebra", "symmetry", "noether", "group_theory", "invariance"]
    ),

    # =========================================================================
    # 3. DIFFERENTIAL GEOMETRY & TOPOLOGY (RIEMANN & PENROSE)
    # =========================================================================
    MathematicalDomain(
        domain_id="math_differential_geometry_riemann",
        name="Differential Geometry, Manifolds & Topology",
        category="Pure Mathematics",
        titans=["Bernhard Riemann", "Henri Poincaré", "Grigori Perelman", "Roger Penrose"],
        core_theorems_and_equations=[
            "Riemannian Metric Tensor: ds^2 = g_mu_nu * dx^mu * dx^nu",
            "Riemann Curvature Tensor: R^rho_sigma_mu_nu measuring tidal deformation and non-Euclidean geometry",
            "Poincaré Conjecture (Perelman's Ricci Flow Theorem): Every simply connected, closed 3-manifold is homeomorphic to the 3-sphere"
        ],
        intuitive_breakthrough=(
            "Space itself is dynamic and deformable. Geometry is not an inert stage, but an active participant "
            "shaped by energy, matter, and topological constraints."
        ),
        reasoning_power_for_ai=(
            "Enables topological data analysis and non-Euclidean manifold learning in latent vector spaces (ChromaDB embeddings)."
        ),
        tags=["geometry", "manifolds", "topology", "curvature", "riemann"]
    ),

    # =========================================================================
    # 4. BLACK HOLE THERMODYNAMICS & SINGULARITIES (HAWKING & PENROSE)
    # =========================================================================
    MathematicalDomain(
        domain_id="math_cosmology_hawking_penrose",
        name="Cosmological Physics & Black Hole Thermodynamics",
        category="Theoretical Physics & Cosmology",
        titans=["Stephen Hawking", "Roger Penrose", "Albert Einstein", "Jacob Bekenstein"],
        core_theorems_and_equations=[
            "Einstein Field Equations: G_mu_nu + Lambda * g_mu_nu = (8 * pi * G / c^4) * T_mu_nu",
            "Hawking Radiation Temperature: T_H = (hbar * c^3) / (8 * pi * G * M * k_B)",
            "Bekenstein-Hawking Black Hole Entropy: S_BH = (k_B * c^3 * A) / (4 * G * hbar)",
            "Penrose-Hawking Singularity Theorems: Under general energy conditions, gravitational collapse inescapably terminates in a spacetime singularity"
        ],
        intuitive_breakthrough=(
            "The unification of General Relativity (gravity, spacetime), Quantum Mechanics (hbar), and Thermodynamics (k_B). "
            "Information and entropy are fundamentally encoded on the surface area of horizons, not their volume (Holographic Principle)."
        ),
        reasoning_power_for_ai=(
            "Gives PaiGen high-caliber cosmic perspective. Teaches that information is never truly destroyed, "
            "and that extreme conditions require synthesis of disparate paradigms."
        ),
        tags=["cosmology", "hawking", "penrose", "black_holes", "relativity", "quantum"]
    ),

    # =========================================================================
    # 5. LINEAR ALGEBRA & OPTIMIZATION (EULER, GAUSS, TURING)
    # =========================================================================
    MathematicalDomain(
        domain_id="math_engineering_linear_algebra",
        name="Linear Algebra, Tensor Calculus & Optimization",
        category="Engineering Mathematics",
        titans=["Leonhard Euler", "Carl Friedrich Gauss", "Gene Golub", "Stephen Boyd"],
        core_theorems_and_equations=[
            "Singular Value Decomposition (SVD): A = U * Sigma * V^T (optimal low-rank matrix approximation)",
            "Eigenvalue Decomposition: A * v = lambda * v (principal axes of variance and invariant directions)",
            "Convex Optimization & Gradient Dynamics: min f(x) s.t. g_i(x) <= 0; Karush-Kuhn-Tucker (KKT) conditions"
        ],
        intuitive_breakthrough=(
            "High-dimensional complexity can be compressed and factored into orthogonal bases, revealing the true "
            "principal energy modes of any system."
        ),
        reasoning_power_for_ai=(
            "Directly powers PaiGen's neural inference, QLoRA weight factorization, dimensionality reduction, "
            "and real-time matrix operations on the GPU."
        ),
        tags=["linear_algebra", "svd", "eigenvalues", "optimization", "gpu"]
    ),

    # =========================================================================
    # 6. INFORMATION THEORY & STOCHASTIC PROCESSES (SHANNON & BAYES)
    # =========================================================================
    MathematicalDomain(
        domain_id="math_information_theory_shannon",
        name="Information Theory, Probability & Stochastic Calculus",
        category="Engineering Mathematics",
        titans=["Claude Shannon", "Thomas Bayes", "Andrey Kolmogorov", "Kiyosi Itô"],
        core_theorems_and_equations=[
            "Shannon Information Entropy: H(X) = - sum(p(x) * log2(p(x)))",
            "Bayes' Theorem: P(A|B) = (P(B|A) * P(A)) / P(B)",
            "Markov Decision Process (MDP) Bellman Equation: V(s) = max_a [ R(s,a) + gamma * sum(P(s'|s,a) * V(s')) ]",
            "Itô's Lemma for Stochastic Differentials: df = (df/dt + mu * df/dx + 0.5 * sigma^2 * d^2f/dx^2) dt + sigma * df/dx * dW_t"
        ],
        intuitive_breakthrough=(
            "Information is the reduction of uncertainty. Probability is not merely frequency, but the degree of belief "
            "dynamically updated in the face of new empirical evidence."
        ),
        reasoning_power_for_ai=(
            "Directly underlies PaiGen's Multi-Armed Bandit reinforcement learning loop, entropy-regulated generation, "
            "and probabilistic emotion modeling."
        ),
        tags=["information_theory", "shannon", "bayes", "reinforcement_learning", "probability"]
    ),

    # =========================================================================
    # 7. FOURIER ANALYSIS & WAVE THEORY (FOURIER & DIRAC)
    # =========================================================================
    MathematicalDomain(
        domain_id="math_fourier_analysis_waves",
        name="Harmonic Analysis, Fourier Transforms & Wavelets",
        category="Engineering Mathematics",
        titans=["Joseph Fourier", "Paul Dirac", "Norbert Wiener"],
        core_theorems_and_equations=[
            "Continuous Fourier Transform: F(omega) = int_{-inf}^{inf} f(t) * exp(-i * omega * t) dt",
            "Fast Fourier Transform (FFT) Algorithm: O(N log N) computational complexity for spectral decomposition",
            "Heisenberg-Gabor Uncertainty Principle in Signal Processing: Delta t * Delta omega >= 1/2"
        ],
        intuitive_breakthrough=(
            "Any arbitrary complex waveform or signal—be it sound, facial landmark movement, or electromagnetic field—"
            "is simply a linear superposition of pure sinusoids."
        ),
        reasoning_power_for_ai=(
            "Enables real-time audio spectrogram analysis for local voice synthesis (Kokoro-ONNX/Piper) "
            "and micro-expression frequency analysis."
        ),
        tags=["fourier", "signal_processing", "waves", "audio", "transforms"]
    ),
]


class MathematicsRegistry:
    @classmethod
    def get_all(cls) -> List[MathematicalDomain]:
        return MATHEMATICS_KNOWLEDGE_BASE

    @classmethod
    def get_by_category(cls, category: str) -> List[MathematicalDomain]:
        return [m for m in MATHEMATICS_KNOWLEDGE_BASE if category.lower() in m.category.lower()]

    @classmethod
    def get_by_titan(cls, titan_name: str) -> List[MathematicalDomain]:
        return [m for m in MATHEMATICS_KNOWLEDGE_BASE if any(titan_name.lower() in t.lower() for t in m.titans)]

