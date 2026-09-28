"""Inference-only reuse scoped to one visual attention invocation."""
def with_qkv_reuse(module, callback):
    original = module.forward
    had_local = 'forward' in module.__dict__
    local = module.__dict__.get('forward')
    cached_input = cached_output = None
    def forward(*args, **kwargs):
        nonlocal cached_input, cached_output
        if len(args) != 1 or kwargs:
            return original(*args, **kwargs)
        if args[0] is cached_input and cached_output is not None:
            return cached_output
        result = original(*args, **kwargs)
        cached_input, cached_output = args[0], result
        return result
    module.forward = forward
    try:
        return callback()
    finally:
        if had_local:
            module.forward = local
        else:
            del module.forward
