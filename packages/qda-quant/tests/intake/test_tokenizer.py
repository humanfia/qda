"""A serialized ByteLevel tokenizer must survive metadata declaring a different tokenizer family."""

import json

from qda_quant.calib.selfgen import _as_text
from qda_quant.engine.calib import load_samples
from qda_quant.intake.tokenizer import load_tokenizer
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


def test_serialized_bytelevel_pipeline_drives_calibration_and_text(tmp_path):
    from qda_quant.calib.sources import FileSource

    text = "Hello world, this is a test.\nNext line."
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator([text], trainers.BpeTrainer(vocab_size=300))
    template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    PreTrainedTokenizerFast(tokenizer_object=backend, chat_template=template).save_pretrained(tmp_path)
    config_path = tmp_path / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config.update(tokenizer_class="LlamaTokenizerFast", legacy=True)
    config_path.write_text(json.dumps(config))

    tok = load_tokenizer(tmp_path)
    expected = backend.encode(text).ids
    assert tok.encode(text, add_special_tokens=False) == expected
    assert tok.decode(expected, clean_up_tokenization_spaces=False) == text
    assert tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False) == text

    calib = tmp_path / "calib.jsonl"
    calib.write_text(json.dumps({"text": text}) + "\n")
    source = FileSource(path=str(calib), samples=1, seqlen=len(expected))
    assert load_samples(source, tmp_path)[0].ids.tolist() == expected
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Hello world,"},
                {"role": "assistant", "content": " this is a test.\nNext line."},
            ]
        }
    ]
    assert _as_text(str(tmp_path), rows, {}) == [{"text": text}]
