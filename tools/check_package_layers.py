"""Check the source dependency order: agent, KL, then quant."""

import ast
import sys
from pathlib import Path

FORBIDDEN = {"qda_quant": {"qda_kl", "qda_agent"}, "qda_kl": {"qda_agent"}}
VENDORED = {"third_party", "vendor", "_vendor"}


def check_tree(root: Path) -> list[str]:
    """Find forbidden imports, including imports inside functions."""
    violations = []
    for package, forbidden in FORBIDDEN.items():
        source = root / "packages" / package.replace("_", "-") / "src" / package
        for path in sorted(source.rglob("*.py")):
            if VENDORED.intersection(path.relative_to(source).parts):
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules = [node.module or ""]
                else:
                    continue
                for module in modules:
                    if module.split(".", 1)[0] in forbidden:
                        violations.append(
                            f"{path.relative_to(root)}:{node.lineno}: {package} imports {module}"
                        )
    return violations


def main() -> int:
    """Report source imports that violate package layering."""
    root = Path(__file__).resolve().parents[1]
    violations = check_tree(root)
    if violations:
        print("Package layering check failed:", file=sys.stderr)
        print("\n".join(violations), file=sys.stderr)
        return 1
    print("Package layering check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
