"""Contract tests for `movieroll.generate`'s argument validation.

These functions run before any weight is loaded and their messages are the user's first contact with a broken
knob, so they are worth pinning exactly: a silently accepted frame count produces a shorter video than the one
that was asked for, which is far worse than a refusal.

CPU only: no checkpoints, no CUDA, no FlashAttention.
"""

import argparse

import pytest

from movieroll.generate import (
    _aspect_bucket_sizes,
    _check_i2v_video_length,
    _check_resolution,
    _check_vae_stream_decode_max_chunk_size,
    _check_video_length,
    _closest_aspect_bucket,
    str_to_bool,
)
from movieroll.commons.profiles import I2V_BUCKET_BASE_SIZE, I2V_BUCKET_MAX_RATIO, I2V_BUCKET_PATCH_SIZE


# ----------------------------------------------------------------------------------------------
# --height / --width
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("height,width", [(480, 864), (864, 480), (16, 16), (640, 640)])
def test_resolution_accepts_positive_multiples_of_16(height, width):
    _check_resolution(height, width)


@pytest.mark.parametrize("height,width", [(480, 865), (0, 864), (-16, 864), (480, 0), (15, 16)])
def test_resolution_rejects_non_multiples_and_non_positives(height, width):
    with pytest.raises(ValueError):
        _check_resolution(height, width)


def test_resolution_rejects_bool_and_float():
    """`True` is an int in Python and 1 is not a multiple of 16, but the reject must be about the type."""
    with pytest.raises(ValueError):
        _check_resolution(True, 864)
    with pytest.raises(ValueError):
        _check_resolution(480.0, 864)


def test_resolution_error_names_the_offending_flag():
    with pytest.raises(ValueError) as excinfo:
        _check_resolution(480, 865)
    assert "--width" in str(excinfo.value)


# ----------------------------------------------------------------------------------------------
# --video-length
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("frame_count", [17, 33, 241, 721, 1441])
def test_video_length_accepts_4n1_with_whole_chunks(frame_count):
    # chunk_size 4: latent frames must be 1 modulo 4, so 17 -> 5, 241 -> 61, 1441 -> 361
    _check_video_length(frame_count, chunk_size=4, sampling_mode="base")


@pytest.mark.parametrize("frame_count", [18, 21, 0, -1])
def test_video_length_rejects_broken_frame_counts(frame_count):
    with pytest.raises(ValueError):
        _check_video_length(frame_count, chunk_size=4, sampling_mode="base")


def test_video_length_suggests_the_nearest_valid_values():
    """The message has to be actionable, otherwise users round in the wrong direction."""
    with pytest.raises(ValueError) as excinfo:
        _check_video_length(21, chunk_size=4, sampling_mode="base")
    message = str(excinfo.value)
    assert "17" in message and "33" in message


def test_distilled_needs_a_whole_first_chunk():
    # latent 1 fits the base chain but the distilled chain needs the input chunk plus one generated chunk
    _check_video_length(1, chunk_size=4, sampling_mode="base")
    with pytest.raises(ValueError):
        _check_video_length(1, chunk_size=4, sampling_mode="distilled")
    _check_video_length(17, chunk_size=4, sampling_mode="distilled")


# ----------------------------------------------------------------------------------------------
# image-to-video length
# ----------------------------------------------------------------------------------------------

def test_i2v_needs_one_generated_chunk_after_the_input_frame():
    _check_i2v_video_length(17, chunk_size=4)
    for frame_count in (1, 13):
        with pytest.raises(ValueError):
            _check_i2v_video_length(frame_count, chunk_size=4)


# ----------------------------------------------------------------------------------------------
# --vae-stream-decode-max-chunk-size
# ----------------------------------------------------------------------------------------------

def test_decode_chunk_size_distilled_reads_with_int():
    _check_vae_stream_decode_max_chunk_size(4, "distilled")
    _check_vae_stream_decode_max_chunk_size("4", "distilled")   # int("4") works, on purpose
    with pytest.raises(ValueError):
        _check_vae_stream_decode_max_chunk_size(None, "distilled")
    with pytest.raises(ValueError):
        _check_vae_stream_decode_max_chunk_size("four", "distilled")


def test_decode_chunk_size_base_takes_none_or_a_positive_int():
    _check_vae_stream_decode_max_chunk_size(None, "base")
    _check_vae_stream_decode_max_chunk_size(4, "base")
    for value in (0, -1, True, "4", 4.0):
        with pytest.raises(ValueError):
            _check_vae_stream_decode_max_chunk_size(value, "base")


# ----------------------------------------------------------------------------------------------
# boolean flag parsing
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, "true", "1", "yes", "on", "TRUE", " True "])
def test_str_to_bool_truthy(value):
    assert str_to_bool(value) is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", "FALSE"])
def test_str_to_bool_falsy(value):
    assert str_to_bool(value) is False


def test_str_to_bool_rejects_anything_else():
    with pytest.raises(argparse.ArgumentTypeError):
        str_to_bool("maybe")


def test_str_to_bool_passes_a_bool_through():
    assert str_to_bool(True) is True
    assert str_to_bool(False) is False


# ----------------------------------------------------------------------------------------------
# i2v aspect buckets
# ----------------------------------------------------------------------------------------------

def test_aspect_buckets_are_well_formed():
    sizes = _aspect_bucket_sizes()
    assert sizes, "the bucket list must not be empty"
    num_patches = round((I2V_BUCKET_BASE_SIZE / I2V_BUCKET_PATCH_SIZE) ** 2)
    for height, width in sizes:
        assert height % I2V_BUCKET_PATCH_SIZE == 0 and width % I2V_BUCKET_PATCH_SIZE == 0
        assert max(height, width) / min(height, width) <= I2V_BUCKET_MAX_RATIO
        assert (height // I2V_BUCKET_PATCH_SIZE) * (width // I2V_BUCKET_PATCH_SIZE) <= num_patches


@pytest.mark.parametrize("height,width", [(1920, 1080), (1080, 1920), (1000, 1000), (3840, 800), (800, 3840)])
def test_closest_bucket_only_trims_the_longer_side(height, width):
    """The one-sided search is what makes cover-and-centre-crop never invent pixels."""
    sizes = _aspect_bucket_sizes()
    chosen = _closest_aspect_bucket(height, width)
    assert chosen in sizes
    image_ratio = height / width
    bucket_ratio = chosen[0] / chosen[1]
    if image_ratio >= 1:
        assert bucket_ratio <= image_ratio
    else:
        assert bucket_ratio >= image_ratio
