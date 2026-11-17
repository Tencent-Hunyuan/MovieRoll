"""Contract tests for `movieroll.commons.profiles`.

The profiles are not tunables: they are the exact values the released checkpoints were sampled with, and
`generate.py` reads them by key. Two failure modes are worth pinning:

1. **Drift.** A knob renamed or removed in a profile surfaces as a `KeyError` deep inside a rollout.
2. **Leakage.** `load_model_dir` merges a checkpoint's `config.json` on top of the profile; if `get_profile`
   returned the module-level dict instead of a copy, one run's config would silently rewrite the defaults for
   every later run in the same process.

CPU only, no checkpoints: `load_model_dir` only stats the weight file.
"""

import json
import os

import pytest

from movieroll.commons.constants import MODEL_TYPE
from movieroll.commons.profiles import (
    BASE,
    COMMON,
    DISTILLED,
    MODEL_CONFIG_FILE,
    MODEL_WEIGHT_FILE,
    SAMPLING_MODES,
    get_profile,
    load_model_dir,
)

# The knobs each chain's pipeline actually reads; losing one is a runtime KeyError, not a typing error.
BASE_KNOBS = {
    "num_inference_steps", "use_cfg", "cfg_min", "cfg_max", "negative_prompt", "flow_shift",
    "context_noise_enabled", "inference_context_noise",
}
DISTILLED_KNOBS = {"denoising_step_list", "context_noise", "timestep_shift"}
SHARED_KNOBS = {
    "model_type", "height", "width", "video_fps", "text_len", "byt5_max_length",
    "prompt_template", "prompt_template_video",
    "chunk_size", "window_size", "full_history_block_num", "max_num_latent_frames",
    "vae_stream_decode_max_chunk_size", "kv_recompute_enabled", "kv_recompute_threshold",
    "kv_recompute_recent_k",
}


# ----------------------------------------------------------------------------------------------
# get_profile
# ----------------------------------------------------------------------------------------------

def test_sampling_modes_are_the_two_documented_chains():
    assert SAMPLING_MODES == ("base", "distilled")


@pytest.mark.parametrize("mode,own_knobs", [("base", BASE_KNOBS), ("distilled", DISTILLED_KNOBS)])
def test_profile_carries_every_knob_its_chain_reads(mode, own_knobs):
    profile = get_profile(mode)
    missing = (own_knobs | SHARED_KNOBS) - set(profile)
    assert not missing, f"{mode} profile is missing {sorted(missing)}"


def test_profile_merges_common_with_the_chain_block():
    base, distilled = get_profile("base"), get_profile("distilled")
    assert set(base) == set(COMMON) | set(BASE)
    assert set(distilled) == set(COMMON) | set(DISTILLED)


def test_unknown_sampling_mode_lists_the_valid_ones():
    with pytest.raises(ValueError) as excinfo:
        get_profile("turbo")
    message = str(excinfo.value)
    for mode in SAMPLING_MODES:
        assert mode in message


def test_get_profile_returns_a_copy():
    """`load_model_dir` merges a checkpoint config into this dict; it must not reach the module defaults."""
    profile = get_profile("base")
    original = profile["chunk_size"]
    profile["chunk_size"] = 999
    assert get_profile("base")["chunk_size"] == original
    assert COMMON["chunk_size"] == original


def test_distilled_window_matches_its_denoising_ladder():
    """`generate.py` refuses to run when window_size != len(denoising_step_list); the profile must agree."""
    distilled = get_profile("distilled")
    assert len(distilled["denoising_step_list"]) == distilled["window_size"]


def test_streaming_geometry_is_shared_by_both_chains():
    base, distilled = get_profile("base"), get_profile("distilled")
    for key in ("chunk_size", "window_size", "full_history_block_num", "max_num_latent_frames"):
        assert base[key] == distilled[key], f"{key} must not differ between the chains"


# ----------------------------------------------------------------------------------------------
# load_model_dir
# ----------------------------------------------------------------------------------------------

def _model_dir(tmp_path, config, *, weights=True, config_name=MODEL_CONFIG_FILE):
    directory = tmp_path / "ckpt"
    directory.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (directory / config_name).write_text(json.dumps(config), encoding="utf-8")
    if weights:
        (directory / MODEL_WEIGHT_FILE).write_bytes(b"")
    return str(directory)


def test_load_model_dir_applies_the_checkpoint_values(tmp_path):
    directory = _model_dir(tmp_path, {
        "sampling_mode": "base",
        "num_inference_steps": 30,
        "height": 512,
        "width": 512,
    })
    settings = load_model_dir(directory)
    assert settings["sampling_mode"] == "base"
    assert settings["num_inference_steps"] == 30
    assert (settings["height"], settings["width"]) == (512, 512)
    # values the file leaves out keep the built-in profile's
    assert settings["chunk_size"] == get_profile("base")["chunk_size"]


def test_load_model_dir_does_not_mutate_the_profile(tmp_path):
    directory = _model_dir(tmp_path, {"sampling_mode": "base", "chunk_size": 8})
    load_model_dir(directory)
    assert get_profile("base")["chunk_size"] != 8


def test_load_model_dir_requires_both_files(tmp_path):
    no_config = _model_dir(tmp_path / "a", None)
    with pytest.raises(FileNotFoundError) as excinfo:
        load_model_dir(no_config)
    assert MODEL_CONFIG_FILE in str(excinfo.value)

    no_weights = _model_dir(tmp_path / "b", {"sampling_mode": "base"}, weights=False)
    with pytest.raises(FileNotFoundError) as excinfo:
        load_model_dir(no_weights)
    assert MODEL_WEIGHT_FILE in str(excinfo.value)


def test_load_model_dir_rejects_an_unknown_sampling_mode(tmp_path):
    directory = _model_dir(tmp_path, {"sampling_mode": "turbo"})
    with pytest.raises(ValueError) as excinfo:
        load_model_dir(directory)
    assert "sampling_mode" in str(excinfo.value)


def test_load_model_dir_rejects_a_missing_sampling_mode(tmp_path):
    directory = _model_dir(tmp_path, {"num_inference_steps": 30})
    with pytest.raises(ValueError):
        load_model_dir(directory)


def test_load_model_dir_rejects_unknown_keys(tmp_path):
    """A typo'd knob must fail here, not be silently ignored and leave the default in force."""
    directory = _model_dir(tmp_path, {"sampling_mode": "base", "num_inference_step": 30})
    with pytest.raises(ValueError) as excinfo:
        load_model_dir(directory)
    assert "num_inference_step" in str(excinfo.value)


def test_load_model_dir_rejects_another_model_type(tmp_path):
    directory = _model_dir(tmp_path, {"sampling_mode": "base", "model_type": "some-other-dit"})
    with pytest.raises(ValueError) as excinfo:
        load_model_dir(directory)
    assert "model_type" in str(excinfo.value)


def test_load_model_dir_accepts_the_released_model_type(tmp_path):
    directory = _model_dir(tmp_path, {"sampling_mode": "distilled", "model_type": MODEL_TYPE})
    assert load_model_dir(directory)["sampling_mode"] == "distilled"
