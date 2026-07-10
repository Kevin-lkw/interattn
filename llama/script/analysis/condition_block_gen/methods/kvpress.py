def build_kvpress_press(method):
    try:
        from kvpress import (
            AdaKVPress,
            ObservedAttentionPress,
            SnapKVPress,
            StreamingLLMPress,
        )
    except ImportError as exc:
        raise ImportError(
            "kvpress is required for KVPress-backed methods. Install it in the "
            "same environment used to run this script."
        ) from exc

    if method.kind == "h2o":
        return ObservedAttentionPress(compression_ratio=method.compression_ratio)
    if method.kind == "kvpress_snapkv":
        return SnapKVPress(
            compression_ratio=method.compression_ratio,
            window_size=method.kvpress_window_size,
            kernel_size=method.kvpress_kernel_size,
        )
    if method.kind == "kvpress_adakv_snapkv":
        return AdaKVPress(
            SnapKVPress(
                compression_ratio=method.compression_ratio,
                window_size=method.kvpress_window_size,
                kernel_size=method.kvpress_kernel_size,
            ),
            alpha_safeguard=method.kvpress_alpha_safeguard,
        )
    if method.kind == "kvpress_streamllm":
        return StreamingLLMPress(
            compression_ratio=method.compression_ratio,
            n_sink=method.kvpress_sink_tokens,
        )
    raise ValueError(f"Not a KVPress method: {method.kind}")
