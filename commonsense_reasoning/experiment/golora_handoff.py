import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1}
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(path: Path, payload: dict) -> None:
    ensure_parent(path)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_journal(path: Path, title: str, lines: list[str]) -> None:
    ensure_parent(path)
    if not path.exists():
        path.write_text("# GO-LoRA Journal\n\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"## {utc_now()} {title}\n")
        for line in lines:
            fh.write(f"- {line}\n")
        fh.write("\n")


def write_next_step(path: Path, lines: list[str]) -> None:
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def common_state(args: argparse.Namespace, status: str, last_event: str) -> dict:
    state = load_state(args.state_path)
    state.update(
        {
            "schema_version": 1,
            "run_id": args.run_id,
            "status": status,
            "tmux_session": args.tmux_session,
            "gpu": args.gpu,
            "target_score": args.target_score,
            "updated_at": utc_now(),
            "last_event": last_event,
        }
    )
    return state


def handle_start_run(args: argparse.Namespace) -> None:
    state = common_state(args, "starting", "start_run")
    state["command"] = args.command
    write_state(args.state_path, state)
    append_journal(
        args.journal_path,
        "Run Started",
        [
            f"run_id={args.run_id}",
            f"tmux_session={args.tmux_session}",
            f"gpu={args.gpu}",
            f"target_score={args.target_score}",
            f"command={args.command}",
        ],
    )
    write_next_step(
        args.next_step_path,
        [
            "# GO-LoRA Next Step",
            "",
            f"Status: starting",
            f"Run ID: {args.run_id}",
            f"Tmux Session: {args.tmux_session}",
            "",
            "Resume commands:",
            f"cd {args.repo_root}",
            f"tmux attach -t {args.tmux_session}",
            "",
            "After attaching, watch the active training/eval logs in tmux.",
        ],
    )


def handle_start_variant(args: argparse.Namespace) -> None:
    state = common_state(args, "running_variant", "start_variant")
    state["current_variant"] = args.variant
    write_state(args.state_path, state)
    append_journal(
        args.journal_path,
        "Variant Started",
        [
            f"variant={args.variant}",
            f"run_id={args.run_id}",
            f"tmux_session={args.tmux_session}",
            f"gpu={args.gpu}",
        ],
    )
    write_next_step(
        args.next_step_path,
        [
            "# GO-LoRA Next Step",
            "",
            "Status: running_variant",
            f"Current variant: {args.variant}",
            f"Run ID: {args.run_id}",
            "",
            "Resume commands:",
            f"cd {args.repo_root}",
            f"tmux attach -t {args.tmux_session}",
            "",
            "Read these files first in a new Codex session:",
            f"- {args.state_path}",
            f"- {args.journal_path}",
            f"- {args.next_step_path}",
        ],
    )


def handle_finish_variant(args: argparse.Namespace) -> None:
    state = common_state(args, "variant_finished", "finish_variant")
    state["current_variant"] = args.variant
    state["last_variant"] = {
        "name": args.variant,
        "average": args.average,
        "weights_dir": args.weights_dir,
        "decision": args.decision,
        "status": args.variant_status,
    }
    write_state(args.state_path, state)
    append_journal(
        args.journal_path,
        "Variant Finished",
        [
            f"variant={args.variant}",
            f"average={args.average}",
            f"weights_dir={args.weights_dir}",
            f"decision={args.decision}",
            f"status={args.variant_status}",
        ],
    )
    write_next_step(
        args.next_step_path,
        [
            "# GO-LoRA Next Step",
            "",
            "Status: variant_finished",
            f"Last variant: {args.variant}",
            f"Average: {args.average}",
            f"Decision: {args.decision}",
            "",
            "Resume commands:",
            f"cd {args.repo_root}",
            f"tmux attach -t {args.tmux_session}",
            "",
            "Then read the latest journal entry and continue with the next formal variant.",
        ],
    )


def handle_finish_run(args: argparse.Namespace) -> None:
    state = common_state(args, args.run_status, "finish_run")
    state["best_variant"] = args.best_variant
    state["best_average"] = args.best_average
    state["best_weights_dir"] = args.best_weights_dir
    state["decision"] = args.decision
    write_state(args.state_path, state)
    append_journal(
        args.journal_path,
        "Run Finished",
        [
            f"run_status={args.run_status}",
            f"best_variant={args.best_variant}",
            f"best_average={args.best_average}",
            f"best_weights_dir={args.best_weights_dir}",
            f"decision={args.decision}",
        ],
    )
    write_next_step(
        args.next_step_path,
        [
            "# GO-LoRA Next Step",
            "",
            f"Status: {args.run_status}",
            f"Best variant: {args.best_variant}",
            f"Best average: {args.best_average}",
            "",
            "Read these files first in a new Codex session:",
            f"- {args.state_path}",
            f"- {args.journal_path}",
            f"- {args.next_step_path}",
            "",
            f"Decision: {args.decision}",
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--state-path", type=Path, required=True)
        subparser.add_argument("--journal-path", type=Path, required=True)
        subparser.add_argument("--next-step-path", type=Path, required=True)
        subparser.add_argument("--run-id", required=True)
        subparser.add_argument("--tmux-session", required=True)
        subparser.add_argument("--gpu", required=True)
        subparser.add_argument("--target-score", required=True)
        subparser.add_argument("--repo-root", default=".")

    start_run = subparsers.add_parser("start-run")
    add_common(start_run)
    start_run.add_argument("--command", required=True)
    start_run.set_defaults(func=handle_start_run)

    start_variant = subparsers.add_parser("start-variant")
    add_common(start_variant)
    start_variant.add_argument("--variant", required=True)
    start_variant.set_defaults(func=handle_start_variant)

    finish_variant = subparsers.add_parser("finish-variant")
    add_common(finish_variant)
    finish_variant.add_argument("--variant", required=True)
    finish_variant.add_argument("--average", required=True)
    finish_variant.add_argument("--weights-dir", required=True)
    finish_variant.add_argument("--decision", required=True)
    finish_variant.add_argument("--variant-status", required=True)
    finish_variant.set_defaults(func=handle_finish_variant)

    finish_run = subparsers.add_parser("finish-run")
    add_common(finish_run)
    finish_run.add_argument("--run-status", required=True)
    finish_run.add_argument("--best-variant", default="")
    finish_run.add_argument("--best-average", default="")
    finish_run.add_argument("--best-weights-dir", default="")
    finish_run.add_argument("--decision", required=True)
    finish_run.set_defaults(func=handle_finish_run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
