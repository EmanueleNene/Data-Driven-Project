"""
Deviatoric Stress/Strain Utilities for 3D Viscoelastic Modeling

For isotropic viscoelastic materials, the deviatoric (shape-changing) stress
components are what drive viscous flow. The Maxwell model applies to deviatoric
components:

    dS_ij/dt = 2G * dε_ij^dev/dt - (2G/η) * S_ij

where:
    S_ij = deviatoric stress tensor
    ε_ij^dev = deviatoric strain rate tensor
    G = shear modulus = E / (2*(1+ν))
    η = viscosity

For an isotropic material, E ≈ 2G (assuming ν ≈ 0 for simplicity),
so the form matches the 1D Maxwell equation.

Author: Generated for DataDrivenProject  
Date: 2025-12-02
"""

import numpy as np


def deviatoric_stress(stress_tensor):
    """
    Compute deviatoric stress from total stress tensor.
    
    Deviatoric stress: S = σ - (trace(σ)/3) * I
    
    This removes the hydrostatic (volumetric) component, leaving only
    the shape-changing (deviatoric) part.
    
    Parameters
    ----------
    stress_tensor : array_like
        Stress components [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_xz]
        Can be:
        - 1D array (6,) for single time point
        - 2D array (n_times, 6) for time series
    
    Returns
    -------
    dev_stress : ndarray
        Deviatoric stress [S_xx, S_yy, S_zz, S_xy, S_yz, S_xz]
        Same shape as input
    
    Examples
    --------
    >>> # Hydrostatic stress has zero deviatoric part
    >>> stress = [100, 100, 100, 0, 0, 0]
    >>> deviatoric_stress(stress)
    array([0., 0., 0., 0., 0., 0.])
    
    >>> # Pure shear is already deviatoric
    >>> stress = [0, 0, 0, 50, 0, 0]
    >>> deviatoric_stress(stress)
    array([ 0.,  0.,  0., 50.,  0.,  0.])
    """
    stress_tensor = np.asarray(stress_tensor)
    
    if stress_tensor.ndim == 1:
        # Single time point
        sxx, syy, szz, txy, tyz, txz = stress_tensor
        
        # Mean stress (hydrostatic component)
        p = (sxx + syy + szz) / 3.0
        
        # Deviatoric normal stresses
        Sxx = sxx - p
        Syy = syy - p
        Szz = szz - p
        
        # Shear stresses are already deviatoric
        Sxy = txy
        Syz = tyz
        Sxz = txz
        
        return np.array([Sxx, Syy, Szz, Sxy, Syz, Sxz])
    
    elif stress_tensor.ndim == 2:
        # Time series
        sxx = stress_tensor[:, 0]
        syy = stress_tensor[:, 1]
        szz = stress_tensor[:, 2]
        txy = stress_tensor[:, 3]
        tyz = stress_tensor[:, 4]
        txz = stress_tensor[:, 5]
        
        # Mean stress
        p = (sxx + syy + szz) / 3.0
        
        # Deviatoric components
        Sxx = sxx - p
        Syy = syy - p
        Szz = szz - p
        Sxy = txy
        Syz = tyz
        Sxz = txz
        
        return np.column_stack([Sxx, Syy, Szz, Sxy, Syz, Sxz])
    
    else:
        raise ValueError(f"Expected 1D or 2D array, got {stress_tensor.ndim}D")


def deviatoric_strain_rate(strain_rate_tensor):
    """
    Compute deviatoric strain rate from total strain rate tensor.
    
    Deviatoric strain rate: ε̇^dev = ε̇ - (trace(ε̇)/3) * I
    
    Parameters
    ----------
    strain_rate_tensor : array_like
        Strain rate [ε̇_xx, ε̇_yy, ε̇_zz, γ̇_xy, γ̇_yz, γ̇_xz]
        where γ̇_ij are engineering shear strain rates (= 2*ε̇_ij)
        Can be:
        - 1D array (6,) for single time point  
        - 2D array (n_times, 6) for time series
    
    Returns
    -------
    dev_strain_rate : ndarray
        Deviatoric strain rate [ε̇^dev_xx, ε̇^dev_yy, ε̇^dev_zz, γ̇_xy, γ̇_yz, γ̇_xz]
        Same shape as input
        
    Note
    ----
    Shear strain rates (γ̇_xy, etc.) are already deviatoric, so they remain unchanged.
    """
    strain_rate_tensor = np.asarray(strain_rate_tensor)
    
    if strain_rate_tensor.ndim == 1:
        # Single time point
        exx, eyy, ezz, gxy, gyz, gxz = strain_rate_tensor
        
        # Volumetric strain rate
        ev = (exx + eyy + ezz) / 3.0
        
        # Deviatoric normal strain rates
        exx_dev = exx - ev
        eyy_dev = eyy - ev
        ezz_dev = ezz - ev
        
        # Engineering shear strain rates are already deviatoric
        gxy_dev = gxy
        gyz_dev = gyz
        gxz_dev = gxz
        
        return np.array([exx_dev, eyy_dev, ezz_dev, gxy_dev, gyz_dev, gxz_dev])
    
    elif strain_rate_tensor.ndim == 2:
        # Time series
        exx = strain_rate_tensor[:, 0]
        eyy = strain_rate_tensor[:, 1]
        ezz = strain_rate_tensor[:, 2]
        gxy = strain_rate_tensor[:, 3]
        gyz = strain_rate_tensor[:, 4]
        gxz = strain_rate_tensor[:, 5]
        
        # Volumetric strain rate
        ev = (exx + eyy + ezz) / 3.0
        
        # Deviatoric components
        exx_dev = exx - ev
        eyy_dev = eyy - ev
        ezz_dev = ezz - ev
        gxy_dev = gxy
        gyz_dev = gyz
        gxz_dev = gxz
        
        return np.column_stack([exx_dev, eyy_dev, ezz_dev, gxy_dev, gyz_dev, gxz_dev])
    
    else:
        raise ValueError(f"Expected 1D or 2D array, got {strain_rate_tensor.ndim}D")


def test_deviatoric():
    """Test deviatoric decomposition with known cases."""
    print("=" * 60)
    print("Testing Deviatoric Decomposition")
    print("=" * 60)
    
    # Test 1: Hydrostatic stress
    print("\nTest 1: Hydrostatic Stress (pure volumetric)")
    stress_hydro = np.array([100.0, 100.0, 100.0, 0, 0, 0])
    dev_stress = deviatoric_stress(stress_hydro)
    print(f"  Input: σ = {stress_hydro}")
    print(f"  Deviatoric: S = {dev_stress}")
    print(f"  Expected: [0, 0, 0, 0, 0, 0]")
    print(f"  ✓ Pass" if np.allclose(dev_stress, 0) else "  ✗ Fail")
    
    # Test 2: Pure shear (already deviatoric)
    print("\nTest 2: Pure Shear (already deviatoric)")
    stress_shear = np.array([0, 0, 0, 50.0, 0, 0])
    dev_stress = deviatoric_stress(stress_shear)
    print(f"  Input: σ = {stress_shear}")
    print(f"  Deviatoric: S = {dev_stress}")
    print(f"  Expected: [0, 0, 0, 50, 0, 0]")
    print(f"  ✓ Pass" if np.allclose(dev_stress, stress_shear) else "  ✗ Fail")
    
    # Test 3: Uniaxial tension
    print("\nTest 3: Uniaxial Tension σ_xx = 100 MPa")
    stress_uniax = np.array([100.0, 0, 0, 0, 0, 0])
    dev_stress = deviatoric_stress(stress_uniax)
    # Deviatoric: S_xx = 100 - 100/3 = 66.67, S_yy = S_zz = -33.33
    expected = np.array([100 - 100/3, -100/3, -100/3, 0, 0, 0])
    print(f"  Input: σ = {stress_uniax}")
    print(f"  Deviatoric: S = {dev_stress}")
    print(f"  Expected: {expected}")
    print(f"  ✓ Pass" if np.allclose(dev_stress, expected) else "  ✗ Fail")
    
    # Test 4: Volumetric strain rate removed
    print("\nTest 4: Deviatoric Strain Rate")
    eps_dot = np.array([0.01, 0.005, 0.003, 0, 0, 0])
    eps_dot_dev = deviatoric_strain_rate(eps_dot)
    ev = (0.01 + 0.005 + 0.003) / 3.0
    expected = np.array([0.01 - ev, 0.005 - ev, 0.003 - ev, 0, 0, 0])
    print(f"  Input: ε̇ = {eps_dot}")
    print(f"  Deviatoric: ε̇^dev = {eps_dot_dev}")
    print(f"  Expected: {expected}")
    print(f"  ✓ Pass" if np.allclose(eps_dot_dev, expected) else "  ✗ Fail")
    
    # Test 5: Trace of deviatoric is zero
    print("\nTest 5: Trace Property")
    stress_random = np.array([50, 30, 20, 10, 5, 3])
    dev_stress = deviatoric_stress(stress_random)
    trace = dev_stress[0] + dev_stress[1] + dev_stress[2]
    print(f"  Input: σ = {stress_random}")
    print(f"  Deviatoric: S = {dev_stress}")
    print(f"  Trace(S): {trace:.6f} (should be 0)")
    print(f"  ✓ Pass" if np.abs(trace) < 1e-10 else "  ✗ Fail")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_deviatoric()
