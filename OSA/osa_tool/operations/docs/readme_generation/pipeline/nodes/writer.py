"""Write the final README to disk and emit operation events."""

import os

from OSA.osa_tool.core.models.event import EventKind, OperationEvent
from OSA.osa_tool.operations.docs.readme_generation.pipeline.state import ReadmeState
from OSA.osa_tool.operations.docs.readme_generation.readme_utils import (
    clean_code_block_indents,
    remove_extra_blank_lines,
    save_sections,
)
from OSA.osa_tool.utils.logger import logger
from OSA.osa_tool.utils.utils import parse_folder_name


def _resolve_target_path(state: ReadmeState) -> str:
    """Return the absolute path for the README file to write."""
    if state.file_to_save:
        return state.file_to_save
    repo_path = os.path.join(os.getcwd(), parse_folder_name(state.repo_url))
    return os.path.join(repo_path, "README.md")


def writer_node(state: ReadmeState) -> dict:
    """Write final README to disk, emit events."""
    logger.info("[Writer] Writing README.md to disk...")

    file_to_save = _resolve_target_path(state)
    output_dir = os.path.dirname(file_to_save)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    readme_content = state.readme_draft or ""
    readme_content = clean_code_block_indents(readme_content)

    save_sections(readme_content, file_to_save)
    remove_extra_blank_lines(file_to_save)

    events = list(state.events)
    events.append(OperationEvent(kind=EventKind.GENERATED, target="README.md"))
    if state.refinement_cycles > 0:
        events.append(OperationEvent(kind=EventKind.REFINED, target="README.md"))

    logger.info("[Writer] README.md written to %s", file_to_save)
    logger.debug("[Writer] State after node: %s", state)
    return {
        "readme_final": readme_content,
        "events": events,
    }
