"""
Von Mises Utilities for 3D Stress and Strain Rate Conversions

This module provides functions to compute von Mises equivalent stress and 
equivalent strain rate from 3D tensor components. These scalar equivalents 
allow us to use 1D constitutive modeling approaches (like SINDy) on 3D data.

Author: Generated for DataDrivenProject
Date: 2025-12-02
"""

import numpy as np


def von_mises_stress(stress_tensor):
    """
    Compute von Mises equivalent stress from 3D stress tensor.
    
    For a stress tensor with components [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_xz],
    the von Mises stress is:
    
    σ_eq = √[(1/2) * ((σ_xx - σ_yy)² + (σ_yy - σ_zz)² + (σ_zz - σ_xx)² 
                      + 6(τ_xy² + τ_yz² + τ_xz²))]
    
    Parameters
    ----------
    stress_tensor : array_like
        Stress components in Voigt notation: [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_xz]
        Can be:
        - 1D array of shape (6,) for a single time point
        - 2D array of shape (n_times, 6) for time series
        - 1D array of shape (3,) for principal stresses [σ_1, σ_2, σ_3]
    
    Returns
    -------
    sigma_eq : float or ndarray
        Von Mises equivalent stress (scalar or time series)
    
    Examples
    --------
    >>> # Uniaxial tension: σ_xx = 100 MPa, others = 0
    >>> stress = [100, 0, 0, 0, 0, 0]
    >>> von_mises_stress(stress)
    100.0
    
    >>> # Pure shear: τ_xy = 50 MPa, others = 0
    >>> stress = [0, 0, 0, 50, 0, 0]
    >>> von_mises_stress(stress)
    86.60254037844387  # = √3 * 50
    """
    stress_tensor = np.asarray(stress_tensor)
    
    # Handle different input shapes
    if stress_tensor.ndim == 1:
        if len(stress_tensor) == 6:
            # Full Voigt notation
            sxx, syy, szz, txy, tyz, txz = stress_tensor
        elif len(stress_tensor) == 3:
            # Principal stresses
            s1, s2, s3 = stress_tensor
            return np.sqrt(0.5 * ((s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2))
        else:
            raise ValueError(f"Expected 3 or 6 components, got {len(stress_tensor)}")
    
    elif stress_tensor.ndim == 2:
        # Time series: shape (n_times, 6)
        if stress_tensor.shape[1] == 6:
            sxx = stress_tensor[:, 0]
            syy = stress_tensor[:, 1]
            szz = stress_tensor[:, 2]
            txy = stress_tensor[:, 3]
            tyz = stress_tensor[:, 4]
            txz = stress_tensor[:, 5]
        elif stress_tensor.shape[1] == 3:
            # Principal stresses time series
            s1 = stress_tensor[:, 0]
            s2 = stress_tensor[:, 1]
            s3 = stress_tensor[:, 2]
            return np.sqrt(0.5 * ((s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2))
        else:
            raise ValueError(f"Expected shape (n, 3) or (n, 6), got {stress_tensor.shape}")
    else:
        raise ValueError(f"Expected 1D or 2D array, got {stress_tensor.ndim}D")
    
    # Von Mises formula
    sigma_eq = np.sqrt(0.5 * (
        (sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 +
        6 * (txy**2 + tyz**2 + txz**2)
    ))
    
    return sigma_eq


def von_mises_strain_rate(strain_rate_tensor):
    """
    Compute von Mises equivalent strain rate from 3D strain rate tensor.
    
    Uses the TOTAL strain rate formulation:
    
    ε̇_eq = √[(2/3) * (ε̇_xx² + ε̇_yy² + ε̇_zz² + (1/2)(γ̇_xy² + γ̇_yz² + γ̇_xz²))]
    
    Note: γ̇_ij = 2*ε̇_ij for engineering shear strain rate
    
    Parameters
    ----------
    strain_rate_tensor : array_like
        Strain rate components in Voigt notation: 
        [ε̇_xx, ε̇_yy, ε̇_zz, γ̇_xy, γ̇_yz, γ̇_xz]
        where γ̇_ij are engineering shear strain rates
        Can be:
        - 1D array of shape (6,) for a single time point
        - 2D array of shape (n_times, 6) for time series
    
    Returns
    -------
    eps_dot_eq : float or ndarray
        Von Mises equivalent strain rate (scalar or time series)
    
    Examples
    --------
    >>> # Uniaxial strain rate: ε̇_xx = 0.01/s, others = 0
    >>> eps_dot = [0.01, 0, 0, 0, 0, 0]
    >>> von_mises_strain_rate(eps_dot)
    0.008164965809277261  # ≈ ε̇_xx * √(2/3)
    """
    strain_rate_tensor = np.asarray(strain_rate_tensor)
    
    # Handle different input shapes
    if strain_rate_tensor.ndim == 1:
        if len(strain_rate_tensor) != 6:
            raise ValueError(f"Expected 6 components, got {len(strain_rate_tensor)}")
        exx, eyy, ezz, gxy, gyz, gxz = strain_rate_tensor
    
    elif strain_rate_tensor.ndim == 2:
        if strain_rate_tensor.shape[1] != 6:
            raise ValueError(f"Expected shape (n, 6), got {strain_rate_tensor.shape}")
        exx = strain_rate_tensor[:, 0]
        eyy = strain_rate_tensor[:, 1]
        ezz = strain_rate_tensor[:, 2]
        gxy = strain_rate_tensor[:, 3]
        gyz = strain_rate_tensor[:, 4]
        gxz = strain_rate_tensor[:, 5]
    else:
        raise ValueError(f"Expected 1D or 2D array, got {strain_rate_tensor.ndim}D")
    
    # Von Mises strain rate formula (total strain rate)
    # Note: engineering shear strain γ = 2ε, so γ̇²/4 = ε̇²
    eps_dot_eq = np.sqrt((2.0/3.0) * (
        exx**2 + eyy**2 + ezz**2 +
        0.5 * (gxy**2 + gyz**2 + gxz**2)
    ))
    
    return eps_dot_eq


def validate_tensor_shape(tensor, expected_components, name="tensor"):
    """
    Validate that a tensor has the expected shape.
    
    Parameters
    ----------
    tensor : array_like
        The tensor to validate
    expected_components : int
        Expected number of components (3 or 6)
    name : str, optional
        Name of the tensor for error messages
    
    Raises
    ------
    ValueError
        If tensor shape is invalid
    """
    tensor = np.asarray(tensor)
    
    if tensor.ndim == 1:
        if len(tensor) != expected_components:
            raise ValueError(
                f"{name} must have {expected_components} components, "
                f"got {len(tensor)}"
            )
    elif tensor.ndim == 2:
        if tensor.shape[1] != expected_components:
            raise ValueError(
                f"{name} must have shape (n, {expected_components}), "
                f"got {tensor.shape}"
            )
    else:
        raise ValueError(f"{name} must be 1D or 2D, got {tensor.ndim}D")


def test_von_mises():
    """
    Test function to verify von Mises calculations with known cases.
    """
    print("=" * 60)
    print("Testing Von Mises Utility Functions")
    print("=" * 60)
    
    # Test 1: Uniaxial tension
    print("\nTest 1: Uniaxial Tension (σ_xx = 100 MPa)")
    stress_uniaxial = np.array([100.0, 0, 0, 0, 0, 0])
    sigma_eq = von_mises_stress(stress_uniaxial)
    print(f"  σ_eq = {sigma_eq:.2f} MPa")
    print(f"  Expected: 100.00 MPa")
    print(f"  ✓ Pass" if np.isclose(sigma_eq, 100.0) else "  ✗ Fail")
    
    # Test 2: Pure shear
    print("\nTest 2: Pure Shear (τ_xy = 50 MPa)")
    stress_shear = np.array([0, 0, 0, 50.0, 0, 0])
    sigma_eq = von_mises_stress(stress_shear)
    expected = np.sqrt(3) * 50.0
    print(f"  σ_eq = {sigma_eq:.2f} MPa")
    print(f"  Expected: {expected:.2f} MPa (√3 * 50)")
    print(f"  ✓ Pass" if np.isclose(sigma_eq, expected) else "  ✗ Fail")
    
    # Test 3: Hydrostatic stress (should give 0)
    print("\nTest 3: Hydrostatic Stress (σ_xx = σ_yy = σ_zz = 100 MPa)")
    stress_hydro = np.array([100.0, 100.0, 100.0, 0, 0, 0])
    sigma_eq = von_mises_stress(stress_hydro)
    print(f"  σ_eq = {sigma_eq:.2f} MPa")
    print(f"  Expected: 0.00 MPa (no deviatoric stress)")
    print(f"  ✓ Pass" if np.isclose(sigma_eq, 0.0) else "  ✗ Fail")
    
    # Test 4: Biaxial tension
    print("\nTest 4: Biaxial Tension (σ_xx = 100, σ_yy = 50 MPa)")
    stress_biaxial = np.array([100.0, 50.0, 0, 0, 0, 0])
    sigma_eq = von_mises_stress(stress_biaxial)
    expected_biaxial = np.sqrt(0.5 * ((100-50)**2 + (50-0)**2 + (0-100)**2))
    print(f"  σ_eq = {sigma_eq:.2f} MPa")
    print(f"  Expected: {expected_biaxial:.2f} MPa")
    print(f"  ✓ Pass" if np.isclose(sigma_eq, expected_biaxial) else "  ✗ Fail")
    
    # Test 5: Time series
    print("\nTest 5: Time Series (varying uniaxial)")
    t = np.linspace(0, 1, 5)
    stress_series = np.zeros((5, 6))
    stress_series[:, 0] = 100 * np.sin(2 * np.pi * t)  # σ_xx varies
    sigma_eq_series = von_mises_stress(stress_series)
    print(f"  t = {t}")
    print(f"  σ_xx = {stress_series[:, 0]}")
    print(f"  σ_eq = {sigma_eq_series}")
    print(f"  ✓ Pass" if np.allclose(np.abs(sigma_eq_series), np.abs(stress_series[:, 0])) else "  ✗ Fail")
    
    # Test 6: Strain rate
    print("\nTest 6: Uniaxial Strain Rate (ε̇_xx = 0.01/s)")
    eps_dot_uniaxial = np.array([0.01, 0, 0, 0, 0, 0])
    eps_dot_eq = von_mises_strain_rate(eps_dot_uniaxial)
    expected_eps = 0.01 * np.sqrt(2.0/3.0)
    print(f"  ε̇_eq = {eps_dot_eq:.6f} /s")
    print(f"  Expected: {expected_eps:.6f} /s (ε̇_xx * √(2/3))")
    print(f"  ✓ Pass" if np.isclose(eps_dot_eq, expected_eps) else "  ✗ Fail")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run tests when module is executed directly
    test_von_mises()
