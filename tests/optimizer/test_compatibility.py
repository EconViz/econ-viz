"""Compatibility tests for legacy optimizer import paths."""


def test_statics_module_reexports_public_helpers():
    from econ_viz.optimizer import statics

    assert statics.__all__ == [
        "ComparativeStatics",
        "comparative_statics",
        "SlutskyMatrix",
        "slutsky_matrix",
    ]
    assert callable(statics.comparative_statics)
    assert callable(statics.slutsky_matrix)
