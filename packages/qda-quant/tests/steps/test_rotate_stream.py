"""One tiny model per declared multimodal family: a forward with an image in the prompt is equal after rotation.

The declaration names a vision merger's Linear as the stream writer; only a forward through the real tower
shows that the rotation reaches the image tokens, so each family's test is the proof of its declaration.
"""

import pytest
import torch
import transformers
from qda_quant import REGISTRY
from qda_quant.arch import describe


def _rotated(m, ids, **kw):
    """The model rotated; returns how much the image changes the logits and the error of the rotation."""
    graph = describe(m, verify=False)
    with torch.no_grad():
        ref = m(input_ids=ids, **kw).logits
        text = m(input_ids=ids).logits
        REGISTRY["rotate"]().apply_model(m, graph, None)
        got = m(input_ids=ids, **kw).logits
    return (ref - text).abs().max().item(), (got - ref).abs().max() / ref.abs().max()


def _prompt(patch_features, merge=2):
    """Ids around one 4x4-patch image, and the pixels and grid both Qwen VL classes take."""
    t, h, w = 1, 4, 4
    n_img = t * h * w // merge**2
    ids = torch.cat(
        [torch.tensor([[7, 8, 122]]), torch.full((1, n_img), 120), torch.tensor([[123, 9, 10, 11]])], 1
    )
    pix = torch.randn(t * h * w, patch_features, generator=torch.Generator().manual_seed(1))
    return ids, pix, torch.tensor([[t, h, w]])


def _ids_around_one_image(n_img=4):
    """The prompt around an image of `n_img` soft tokens."""
    return torch.cat(
        [torch.tensor([[7, 8, 122]]), torch.full((1, n_img), 120), torch.tensor([[123, 9, 10, 11]])], 1
    )


def _tiny_qwen3_vl(deepstack=()):
    text = transformers.Qwen3VLTextConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, vocab_size=128, max_position_embeddings=64,
        tie_word_embeddings=False,
    )  # fmt: skip
    vision = transformers.Qwen3VLVisionConfig(
        depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=128,
        num_position_embeddings=16, spatial_merge_size=2, deepstack_visual_indexes=deepstack,
    )  # fmt: skip
    cfg = transformers.Qwen3VLConfig(
        text_config=text, vision_config=vision, image_token_id=120, video_token_id=121,
        vision_start_token_id=122, vision_end_token_id=123,
    )  # fmt: skip
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return transformers.Qwen3VLForConditionalGeneration._from_config(cfg).eval().to(torch.float64)


@pytest.mark.parametrize("deepstack", [(), (0,)], ids=["plain", "deepstack"])
def test_qwen3_vl_images_land_in_the_rotated_space(deepstack):
    m = _tiny_qwen3_vl(deepstack)
    ids, pix, grid = _prompt(3 * 2 * 16**2)
    touched, error = _rotated(
        m, ids, pixel_values=pix, image_grid_thw=grid, mm_token_type_ids=(ids == 120).int()
    )
    assert touched > 1e-3 and error < 1e-5


def test_rotate_refuses_a_nonlinear_step_after_a_declared_writer():
    """The declaration still names a Linear of the residual width; only running the tower shows the GELU."""
    from qda_quant.step import Refused

    m = _tiny_qwen3_vl((0,))
    m.model.visual.merger.register_forward_hook(lambda mod, args, out: torch.nn.functional.gelu(out))
    with pytest.raises(Refused, match=r"does not rotate the output of model\.visual"):
        REGISTRY["rotate"]().apply_model(m, describe(m, verify=False), None)


def _tiny_qwen2_5_vl():
    text = transformers.Qwen2_5_VLTextConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, vocab_size=128, max_position_embeddings=64,
        tie_word_embeddings=False, rope_scaling={"type": "default", "mrope_section": [8, 4, 4]},
    )  # fmt: skip
    vision = transformers.Qwen2_5_VLVisionConfig(
        depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=128,
        spatial_merge_size=2, fullatt_block_indexes=(), window_size=32, patch_size=8, temporal_patch_size=1,
    )  # fmt: skip
    cfg = transformers.Qwen2_5_VLConfig(
        text_config=text, vision_config=vision, image_token_id=120, video_token_id=121,
        vision_start_token_id=122, vision_end_token_id=123,
    )  # fmt: skip
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return transformers.Qwen2_5_VLForConditionalGeneration._from_config(cfg).eval().to(torch.float64)


def test_qwen2_5_vl_images_land_in_the_rotated_space():
    m = _tiny_qwen2_5_vl()
    ids, pix, grid = _prompt(3 * 8**2, merge=2)
    touched, error = _rotated(
        m, ids, pixel_values=pix, image_grid_thw=grid, mm_token_type_ids=(ids == 120).int()
    )
    assert touched > 1e-3 and error < 1e-5


def _tiny_mistral3():
    text = transformers.MistralConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, vocab_size=128, max_position_embeddings=64,
        tie_word_embeddings=False,
    )  # fmt: skip
    vision = transformers.PixtralVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, head_dim=16,
        patch_size=16, image_size=64,
    )  # fmt: skip
    cfg = transformers.Mistral3Config(
        vision_config=vision, text_config=text, image_token_index=120, tie_word_embeddings=False
    )  # fmt: skip
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return transformers.Mistral3ForConditionalGeneration._from_config(cfg).eval().to(torch.float64)


def test_mistral3_images_land_in_the_rotated_space():
    m = _tiny_mistral3()
    ids = _ids_around_one_image()  # a 64x64 image at patch 16, merged 2x2, is 2x2 tokens
    pix = torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(1))
    touched, error = _rotated(m, ids, pixel_values=pix, image_sizes=torch.tensor([[64, 64]]))
    assert touched > 1e-3 and error < 1e-5


def _tiny_gemma4():
    text = transformers.Gemma4TextConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, vocab_size=128, max_position_embeddings=64,
        vocab_size_per_layer_input=128, hidden_size_per_layer_input=32, tie_word_embeddings=False,
    )  # fmt: skip
    vision = transformers.Gemma4VisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, head_dim=16, patch_size=16, position_embedding_size=64, pooling_kernel_size=2,
    )  # fmt: skip
    cfg = transformers.Gemma4Config(
        text_config=text, vision_config=vision, image_token_id=120, boi_token_id=121, eoi_token_id=122,
        video_token_id=123, tie_word_embeddings=False,
    )  # fmt: skip
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return transformers.Gemma4ForConditionalGeneration._from_config(cfg).eval()


def test_gemma4_declares_its_vision_stream_and_rotate_refuses_its_post_norm():
    from qda_quant.step import Refused

    graph = describe(_tiny_gemma4(), verify=False)
    assert graph.extra["outside"] == ("model.embed_vision", "model.vision_tower")
    assert graph.extra["stream"] == {
        "writers": {
            "model.embed_vision": ("model.embed_vision.embedding_projection",),
            "model.vision_tower": (),
        }
    }
    with pytest.raises(Refused, match=r"post-norm blocks"):
        REGISTRY["rotate"]().check(graph, None, None)
