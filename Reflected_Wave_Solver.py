from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Candidate:
    """One possible solution z = x + i y."""

    z: complex
    stationary_residual: float
    normalized_stationary_residual: float
    magnitude_residual_plus_minus: float
    magnitude_residual_jp_plus: float


@dataclass
class ExtractionResult:
    """Final result of the coefficient-extraction algorithm."""

    z: complex
    A_d: complex
    A_J: complex
    candidates: List[Candidate]
    selected_candidate_index: int
    predicted_R_plus: float
    predicted_R_minus: float
    predicted_R_Jp: float
    circle_1: tuple[float, float, float, float]
    circle_2: tuple[float, float, float, float]
    radical_axis: tuple[float, float, float]


def p_function(omega: float, omega_d: float) -> float:
    """p(omega) = 1 / (omega - omega_d)."""
    denominator = omega - omega_d
    if np.isclose(denominator, 0.0):
        raise ValueError("p(omega) is singular because omega equals omega_d.")
    return 1.0 / denominator


def q_function(omega: float, omega_J: float, kappa_J: float) -> complex:
    """q(omega) = 1 / [kappa_J/2 + i(omega - omega_J)]."""
    gamma = kappa_J / 2.0
    if gamma <= 0.0:
        raise ValueError("kappa_J must be positive.")
    return 1.0 / (gamma + 1j * (omega - omega_J))


def h_function(
    omega: float,
    z: complex,
    omega_d: float,
    omega_J: float,
    kappa_J: float,
) -> complex:
    """h(omega; z) = p(omega) + z q(omega)."""
    return p_function(omega, omega_d) + z * q_function(
        omega, omega_J, kappa_J
    )


def h_prime_function(
    omega: float,
    z: complex,
    omega_d: float,
    omega_J: float,
    kappa_J: float,
) -> complex:
    """h'(omega; z) = -p(omega)^2 - i z q(omega)^2."""
    p = p_function(omega, omega_d)
    q = q_function(omega, omega_J, kappa_J)
    return -(p**2) - 1j * z * (q**2)


def stationary_residual(
    omega_Jp: float,
    z: complex,
    omega_d: float,
    omega_J: float,
    kappa_J: float,
    tolerance: float = 1.0e-15,
) -> tuple[float, float]:
    """Return S = Re[h' h*] and its normalized absolute value."""
    h = h_function(omega_Jp, z, omega_d, omega_J, kappa_J)
    h_prime = h_prime_function(omega_Jp, z, omega_d, omega_J, kappa_J)
    residual = float(np.real(h_prime * np.conjugate(h)))
    normalization = abs(h_prime) * abs(h)
    normalized = (
        abs(residual)
        if normalization <= tolerance
        else abs(residual) / normalization
    )
    return residual, normalized


def circle_coefficients(
    omega_d: float,
    omega_J: float,
    kappa_J: float,
    omega_plus: float,
    omega_minus: float,
    omega_Jp: float,
    R_d: float,
    R_J: float,
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    """Build the two circle equations for x = Re(z), y = Im(z)."""
    if R_d <= 0.0:
        raise ValueError("R_d must be positive.")
    if R_J < 0.0:
        raise ValueError("R_J cannot be negative.")

    p_plus = p_function(omega_plus, omega_d)
    p_minus = p_function(omega_minus, omega_d)
    p_Jp = p_function(omega_Jp, omega_d)

    q_plus = q_function(omega_plus, omega_J, kappa_J)
    q_minus = q_function(omega_minus, omega_J, kappa_J)
    q_Jp = q_function(omega_Jp, omega_J, kappa_J)

    r = R_J / R_d

    a_1 = abs(q_plus) ** 2 - abs(q_minus) ** 2
    b_1 = 2.0 * (p_plus * q_plus.real - p_minus * q_minus.real)
    c_1 = -2.0 * (p_plus * q_plus.imag - p_minus * q_minus.imag)
    d_1 = p_plus**2 - p_minus**2

    a_2 = abs(q_Jp) ** 2 - r**2 * abs(q_plus) ** 2
    b_2 = 2.0 * (p_Jp * q_Jp.real - r**2 * p_plus * q_plus.real)
    c_2 = -2.0 * (p_Jp * q_Jp.imag - r**2 * p_plus * q_plus.imag)
    d_2 = p_Jp**2 - r**2 * p_plus**2

    return (a_1, b_1, c_1, d_1), (a_2, b_2, c_2, d_2)


def remove_duplicate_candidates(
    candidates: List[complex], tolerance: float
) -> List[complex]:
    """Remove numerically duplicated complex roots."""
    unique: List[complex] = []
    for z in candidates:
        duplicate = any(
            abs(z - current) <= tolerance * max(1.0, abs(z), abs(current))
            for current in unique
        )
        if not duplicate:
            unique.append(z)
    return unique


def solve_circle_intersections(
    circle_1: tuple[float, float, float, float],
    circle_2: tuple[float, float, float, float],
    tolerance: float = 1.0e-12,
) -> tuple[List[complex], tuple[float, float, float]]:
    """Find all real intersections and return them as z = x + i y."""
    a_1, b_1, c_1, d_1 = circle_1
    a_2, b_2, c_2, d_2 = circle_2

    L_x = a_2 * b_1 - a_1 * b_2
    L_y = a_2 * c_1 - a_1 * c_2
    L_0 = a_2 * d_1 - a_1 * d_2
    radical_axis = (L_x, L_y, L_0)

    scale = max(1.0, abs(L_x), abs(L_y), abs(L_0))
    axis_tolerance = tolerance * scale
    candidates: List[complex] = []

    if abs(L_y) > axis_tolerance:
        m = -L_x / L_y
        n = -L_0 / L_y
        A_x = a_1 * (1.0 + m**2)
        B_x = 2.0 * a_1 * m * n + b_1 + c_1 * m
        C_x = a_1 * n**2 + c_1 * n + d_1
        coeff_tol = tolerance * max(1.0, abs(A_x), abs(B_x), abs(C_x))

        if abs(A_x) <= coeff_tol:
            if abs(B_x) <= coeff_tol:
                raise ValueError("Degenerate equations do not determine isolated z values.")
            x = -C_x / B_x
            candidates.append(complex(x, m * x + n))
        else:
            discriminant = B_x**2 - 4.0 * A_x * C_x
            disc_tol = tolerance * max(1.0, B_x**2, abs(4.0 * A_x * C_x))
            if discriminant < -disc_tol:
                raise ValueError(
                    "No exact real intersection. Check the data or use least squares."
                )
            root = np.sqrt(max(discriminant, 0.0))
            for sign in (1.0, -1.0):
                x = (-B_x + sign * root) / (2.0 * A_x)
                candidates.append(complex(x, m * x + n))

    elif abs(L_x) > axis_tolerance:
        x_0 = -L_0 / L_x
        A_y = a_1
        B_y = c_1
        C_y = a_1 * x_0**2 + b_1 * x_0 + d_1
        coeff_tol = tolerance * max(1.0, abs(A_y), abs(B_y), abs(C_y))

        if abs(A_y) <= coeff_tol:
            if abs(B_y) <= coeff_tol:
                raise ValueError("Degenerate equations do not determine isolated z values.")
            candidates.append(complex(x_0, -C_y / B_y))
        else:
            discriminant = B_y**2 - 4.0 * A_y * C_y
            disc_tol = tolerance * max(1.0, B_y**2, abs(4.0 * A_y * C_y))
            if discriminant < -disc_tol:
                raise ValueError(
                    "No exact real intersection. Check the data or use least squares."
                )
            root = np.sqrt(max(discriminant, 0.0))
            for sign in (1.0, -1.0):
                y = (-B_y + sign * root) / (2.0 * A_y)
                candidates.append(complex(x_0, y))
    else:
        raise ValueError(
            "The magnitude equations are coincident or numerically dependent."
        )

    return remove_duplicate_candidates(candidates, tolerance), radical_axis


def evaluate_candidate(
    z: complex,
    omega_d: float,
    omega_J: float,
    kappa_J: float,
    omega_plus: float,
    omega_minus: float,
    omega_Jp: float,
    R_d: float,
    R_J: float,
) -> Candidate:
    """Calculate the magnitude and stationary residuals for one z."""
    h_plus = h_function(omega_plus, z, omega_d, omega_J, kappa_J)
    h_minus = h_function(omega_minus, z, omega_d, omega_J, kappa_J)
    h_Jp = h_function(omega_Jp, z, omega_d, omega_J, kappa_J)
    r = R_J / R_d

    S, S_normalized = stationary_residual(
        omega_Jp, z, omega_d, omega_J, kappa_J
    )

    return Candidate(
        z=z,
        stationary_residual=S,
        normalized_stationary_residual=S_normalized,
        magnitude_residual_plus_minus=float(abs(h_plus) ** 2 - abs(h_minus) ** 2),
        magnitude_residual_jp_plus=float(
            abs(h_Jp) ** 2 - r**2 * abs(h_plus) ** 2
        ),
    )


def reflected_spectrum(
    omega: float | np.ndarray,
    A_d: complex,
    A_J: complex,
    sigma_vd: float,
    omega_d: float,
    omega_J: float,
    kappa_J: float,
) -> complex | np.ndarray:
    """Evaluate the original complex reflected spectral function."""
    omega_array = np.asarray(omega)
    drive_term = np.conjugate(A_d) / (omega_array - omega_d)
    qubit_term = 1j * np.conjugate(A_J) / (
        kappa_J / 2.0 + 1j * (omega_array - omega_J)
    )
    result = sigma_vd / np.sqrt(2.0) * (drive_term + qubit_term)
    return complex(result) if result.ndim == 0 else result


def extract_qubit_coefficients(
    *,
    omega_d: float,
    omega_J: float,
    kappa_J: float,
    omega_plus: float,
    omega_minus: float,
    omega_Jp: float,
    R_d: float,
    R_J: float,
    sigma_vd: float,
    common_phase: float = 0.0,
    tolerance: float = 1.0e-12,
) -> ExtractionResult:
    """Extract z, A_d, and A_J from the three magnitude conditions."""
    if sigma_vd <= 0.0:
        raise ValueError("sigma_vd must be positive.")

    circle_1, circle_2 = circle_coefficients(
        omega_d, omega_J, kappa_J, omega_plus, omega_minus,
        omega_Jp, R_d, R_J
    )
    z_values, radical_axis = solve_circle_intersections(
        circle_1, circle_2, tolerance
    )

    candidates = [
        evaluate_candidate(
            z, omega_d, omega_J, kappa_J, omega_plus,
            omega_minus, omega_Jp, R_d, R_J
        )
        for z in z_values
    ]
    if not candidates:
        raise RuntimeError("No candidate values of z were found.")

    selected_index = int(
        np.argmin([c.normalized_stationary_residual for c in candidates])
    )
    z = candidates[selected_index].z
    h_plus = h_function(omega_plus, z, omega_d, omega_J, kappa_J)
    if abs(h_plus) <= tolerance:
        raise ValueError("The selected solution has h(omega_plus) = 0.")

    A_d_magnitude = np.sqrt(2.0) * R_d / (sigma_vd * abs(h_plus))
    A_d = A_d_magnitude * np.exp(-1j * common_phase)
    A_J = 1j * np.conjugate(z) * A_d

    def predicted(omega: float) -> float:
        return abs(
            reflected_spectrum(
                omega, A_d, A_J, sigma_vd, omega_d, omega_J, kappa_J
            )
        )

    return ExtractionResult(
        z=z,
        A_d=A_d,
        A_J=A_J,
        candidates=candidates,
        selected_candidate_index=selected_index,
        predicted_R_plus=predicted(omega_plus),
        predicted_R_minus=predicted(omega_minus),
        predicted_R_Jp=predicted(omega_Jp),
        circle_1=circle_1,
        circle_2=circle_2,
        radical_axis=radical_axis,
    )


def print_result(result: ExtractionResult) -> None:
    """Print the solution, candidate residuals, and consistency checks."""
    print("Candidate solutions")
    print("-------------------")
    for index, candidate in enumerate(result.candidates):
        selected = "  <-- selected" if index == result.selected_candidate_index else ""
        print(f"Candidate {index}:")
        print(f"  z = {candidate.z:.12g}")
        print(f"  stationary residual = {candidate.stationary_residual:.6e}")
        print(
            "  normalized stationary residual = "
            f"{candidate.normalized_stationary_residual:.6e}"
        )
        print(
            "  plus/minus magnitude residual = "
            f"{candidate.magnitude_residual_plus_minus:.6e}"
        )
        print(
            "  Jp/plus magnitude residual = "
            f"{candidate.magnitude_residual_jp_plus:.6e}{selected}"
        )
        print()

    print("Selected coefficients")
    print("---------------------")
    print(f"z   = {result.z:.12g}")
    print(f"A_d = {result.A_d:.12g}")
    print(f"A_J = {result.A_J:.12g}")
    print()
    print("Verification")
    print("------------")
    print(f"Predicted |v(omega_plus)|  = {result.predicted_R_plus:.12g}")
    print(f"Predicted |v(omega_minus)| = {result.predicted_R_minus:.12g}")
    print(f"Predicted |v(omega_Jp)|    = {result.predicted_R_Jp:.12g}")


if __name__ == "__main__":
    # Replace these example values with measured data.
    # All frequencies and kappa_J must use the same units.
    omega_d = 13.000
    omega_J = 13.500
    kappa_J = 0.040

    omega_plus = 13.020
    omega_minus = 12.980
    omega_Jp = 13.505

    R_d = 1.000
    R_J = 0.350
    sigma_vd = 1.000

    result = extract_qubit_coefficients(
        omega_d=omega_d,
        omega_J=omega_J,
        kappa_J=kappa_J,
        omega_plus=omega_plus,
        omega_minus=omega_minus,
        omega_Jp=omega_Jp,
        R_d=R_d,
        R_J=R_J,
        sigma_vd=sigma_vd,
        common_phase=0.0,
    )

    print_result(result)
