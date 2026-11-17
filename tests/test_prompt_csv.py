"""Contract tests for `movieroll.utils.prompt_csv.read_prompt_csv`.

Standard library only -- no torch, no CUDA, no checkpoints. These pin the CSV contract that
`movieroll.generate` and the SFT sampler both rely on, including the numeric (not lexicographic)
ordering of `event_*` columns and the "both globals or neither" rule.

Run either way:
    pytest tests/test_prompt_csv.py
    python tests/test_prompt_csv.py
"""

import os

from movieroll.utils.prompt_csv import read_prompt_csv


def _write(tmpdir, name, text):
    path = os.path.join(str(tmpdir), name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def test_event_columns_sort_numerically(tmp_path):
    """event_10 must come after event_2, not before it (lexicographic order would invert them)."""
    csv_path = _write(
        tmp_path, "run/prompts.csv",
        "index,seed,event_1,event_2,event_10\n"
        "0,7,first,second,tenth\n",
    )
    parsed = read_prompt_csv(csv_path)
    assert parsed.event_cols == ["event_1", "event_2", "event_10"]
    assert parsed.event_prompts[0] == ["first", "second", "tenth"]


def test_prompts_column_is_the_first_event(tmp_path):
    """`prompts` mirrors the first event so callers that ignore events still get a usable prompt."""
    csv_path = _write(
        tmp_path, "prompts.csv",
        "index,seed,event_1,event_2\n0,11,a,b\n",
    )
    parsed = read_prompt_csv(csv_path)
    assert parsed.prompts == ["a"]
    assert parsed.event_prompts == [["a", "b"]]


def test_global_prompts_are_joined_when_both_present(tmp_path):
    csv_path = _write(
        tmp_path, "prompts.csv",
        "index,seed,event_1,global_content_prompt,global_quality_prompt\n"
        "0,1,e1,CONTENT,QUALITY\n",
    )
    parsed = read_prompt_csv(csv_path)
    assert parsed.event_prompts[0] == ["CONTENT e1 QUALITY"]


def test_global_prompts_ignored_when_only_one_present(tmp_path):
    """Half a global pair is ignored entirely -- never silently half-applied."""
    only_content = _write(
        tmp_path, "a/prompts.csv",
        "index,seed,event_1,global_content_prompt\n0,1,e1,CONTENT\n",
    )
    assert read_prompt_csv(only_content).event_prompts[0] == ["e1"]

    only_quality = _write(
        tmp_path, "b/prompts.csv",
        "index,seed,event_1,global_quality_prompt\n0,1,e1,QUALITY\n",
    )
    assert read_prompt_csv(only_quality).event_prompts[0] == ["e1"]


def test_global_prompts_never_applied_to_a_prompt_column_csv(tmp_path):
    """A `prompt`-column CSV has no events, so the globals must not be read even if the columns exist."""
    csv_path = _write(
        tmp_path, "prompts.csv",
        "index,seed,prompt,global_content_prompt,global_quality_prompt\n"
        "0,1,solo,CONTENT,QUALITY\n",
    )
    parsed = read_prompt_csv(csv_path)
    assert parsed.event_prompts == [["solo"]]
    assert parsed.prompts == ["solo"]


def test_seed_is_int_and_index_is_kept_as_text(tmp_path):
    csv_path = _write(
        tmp_path, "prompts.csv",
        "index,seed,prompt\n007,930,hello\n",
    )
    parsed = read_prompt_csv(csv_path)
    assert parsed.seeds == [930]
    assert isinstance(parsed.seeds[0], int)
    # `index` is used to build the output file name, so leading zeros must survive.
    assert parsed.indexs == ["007"]


def test_csv_names_are_parent_dir_and_stem(tmp_path):
    csv_path = _write(tmp_path, "batch_a/my_prompts.csv", "index,seed,prompt\n0,1,x\n")
    parsed = read_prompt_csv(csv_path)
    assert parsed.csv_names == [("batch_a", "my_prompts")]


def test_input_image_empty_cell_means_text_to_video(tmp_path):
    csv_path = _write(
        tmp_path, "prompts.csv",
        "index,seed,prompt,input_image\n0,1,a,\n1,2,b,first_frame.png\n",
    )
    parsed = read_prompt_csv(csv_path)
    assert parsed.has_input_image is True
    assert parsed.input_images[0] is None
    # Relative paths resolve against the CSV's own directory, not the process cwd.
    assert parsed.input_images[1] == os.path.join(str(tmp_path), "first_frame.png")


def test_no_input_image_column_reports_false(tmp_path):
    csv_path = _write(tmp_path, "prompts.csv", "index,seed,prompt\n0,1,a\n")
    parsed = read_prompt_csv(csv_path)
    assert parsed.has_input_image is False
    assert parsed.input_images == [None]


def test_short_row_fails_loudly_rather_than_silently(tmp_path):
    """A row missing an event cell currently raises instead of producing a truncated sample.

    Pinned as documented behaviour: silently emitting a 2-event video for a 3-event CSV would be
    worse than a crash. If this is ever turned into a friendly message, update this test -- the
    important part is that it never returns a short row.
    """
    csv_path = _write(
        tmp_path, "prompts.csv",
        "index,seed,event_1,event_2\n0,1,only_first\n",
    )
    try:
        read_prompt_csv(csv_path)
    except (AttributeError, KeyError):
        return
    raise AssertionError("a short row must not silently produce a truncated sample")
