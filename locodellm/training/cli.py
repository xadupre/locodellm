"""Registers training options without loading optional PyTorch dependencies."""

from pathlib import Path


def add_parser(subparsers):
    """Registers the public multi-turn SLM-GRPO training command."""
    parser = subparsers.add_parser("train", help="Train multi-turn ATen export discovery.")
    parser.add_argument("model", help="Transformers model id or local checkpoint.")
    parser.add_argument("--protocol", choices=["slm-grpo"], default="slm-grpo")
    parser.add_argument("--input", required=True, help="Pinned ATen discovery JSONL.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", help="Clean local source at the recorded revision.")
    parser.add_argument("--adapter", help="Warm-start LoRA weights, not optimizer state.")
    parser.add_argument(
        "--lora-layers",
        choices=["all", "last-half"],
        default="all",
        help="Selects decoder layers for a new adapter; saved adapters retain their selection.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-tool-calls", type=int, default=10)
    parser.add_argument("--harness", choices=["python", "bash"], default="bash")
    parser.add_argument("--tool-timeout", type=float, default=30)
    parser.add_argument("--verification-timeout", type=float, default=60)
    parser.add_argument("--max-duration", type=float, default=480)
    parser.add_argument("--final-answer-after", type=float, default=360)
    parser.add_argument("--updates-per-group", type=int, default=1)
    parser.add_argument("--normalize-advantages", action="store_true")
    parser.add_argument("--checkpoint-steps", type=int)
    parser.add_argument("--checkpoint-minutes", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", "-v", type=int, default=0)
    parser.add_argument(
        "--log-memory",
        action="store_true",
        help="Records CUDA memory usage and per-phase peaks in output/memory.jsonl.",
    )
    parser.add_argument(
        "--allow-code-execution",
        action="store_true",
        help="Allows arbitrary generated code; requires a disposable isolated environment.",
    )
    return parser


def run(args):
    """Loads the optional training implementation only when explicitly selected."""
    from locodellm.training.grpo import run

    run(args)
