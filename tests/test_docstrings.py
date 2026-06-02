import imcombiners as imc
from imcombiners import kernels


def test_kernel_docstrings_are_installed_from_shared_templates():
    assert kernels.nanaverage.__doc__ is not None
    assert "Return the NaN-aware weighted average" in kernels.nanaverage.__doc__
    assert "arr : ndarray, shape (N, *spatial)" in kernels.nanaverage.__doc__
    assert "Accepted dtypes are" in kernels.nanaverage.__doc__


def test_rejection_kernel_docstrings_keep_diagnostics_details():
    assert kernels.sigclip.__doc__ is not None
    assert "Sigma-clipping rejection." in kernels.sigclip.__doc__
    assert "revert_on_nkeep : bool" in kernels.sigclip.__doc__
    assert "std : ndarray, shape (*spatial)" in kernels.sigclip.__doc__
    assert "bit ``16``" in kernels.sigclip.__doc__


def test_rejector_docstrings_share_rejection_parameter_language():
    assert imc.SigClip.__doc__ is not None
    assert "Iterative sigma-clipping rejection." in imc.SigClip.__doc__
    assert "revert_on_nkeep : bool" in imc.SigClip.__doc__
    assert "Optional radius in pixels" in imc.SigClip.__doc__


def test_linearclip_is_public_and_documented():
    assert imc.LinearClip.__doc__ is not None
    assert "Center-relative" in imc.LinearClip.__doc__
    assert "low_scale" in imc.LinearClip.__doc__


def test_output_only_reject_kernels_are_public_and_documented():
    expected = [
        "sigclip_mask",
        "sigclip_combine",
        "ccdclip_mask",
        "ccdclip_combine",
        "minmax_mask",
        "minmax_combine",
        "pclip_mask",
        "pclip_combine",
    ]

    for name in expected:
        assert name in kernels.__all__
        func = getattr(kernels, name)
        assert func.__doc__ is not None
        assert not name.startswith("_")


def test_rejection_variant_docstrings_include_public_parameters():
    assert kernels.sigclip_combine.__doc__ is not None
    assert "combine : str" in kernels.sigclip_combine.__doc__
    assert "Output combine method evaluated after rejection" in (
        kernels.sigclip_combine.__doc__
    )
    assert '`"mean"`, `"average"`, `"avg"`, `"median"`, and `"med"`' in (
        kernels.sigclip_combine.__doc__
    )
    assert "combined : ndarray, shape (*spatial)" in kernels.sigclip_combine.__doc__

    assert kernels.minmax_mask_1d.__doc__ is not None
    assert "values : ndarray, shape (N,)" in kernels.minmax_mask_1d.__doc__
    assert "n_min : int or float" in kernels.minmax_mask_1d.__doc__
    assert "mask_rej : ndarray of bool, shape (N,)" in kernels.minmax_mask_1d.__doc__
