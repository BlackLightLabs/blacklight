from blacklight.engine.adapters import adapter as adapter_module


def test_adapter_check_return_true():
    adapter = adapter_module.Adapter()

    assert adapter.check(None)
