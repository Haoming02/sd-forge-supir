"""
Credit: kijai
https://github.com/kijai/ComfyUI-SUPIR/blob/91c0e185810d8784f9fc0d9eb97a26f477d13338/nodes.py#L33

Modified by. Haoming02 to work with Forge
"""

import open_clip


def build_text_model_from_openai_state_dict(state_dict, cast_dtype):
    open_clip.model._build_vision_tower = lambda *args, **kwargs: None

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]

    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")
        )
    )

    text_cfg = open_clip.CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )

    model = open_clip.CLIP(
        embed_dim,
        vision_cfg=None,
        text_cfg=text_cfg,
        quick_gelu=True,
        cast_dtype=cast_dtype,
    )

    model.load_state_dict(state_dict, strict=False)
    model = model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model
