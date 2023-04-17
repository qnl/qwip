from collections import defaultdict
from pathlib import Path

import yaml
from rich import print


class YAMLDumper(yaml.Dumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False):
        return super().increase_indent(flow, False)


def build_api_pages(src_dir: Path, docs_dir: Path) -> dict:
    directory = defaultdict(dict)

    for path in sorted(src_dir.rglob("*.py")):
        module_path = path.relative_to(src_dir).with_suffix("")
        doc_path = path.relative_to(src_dir).with_suffix(".md")
        full_doc_path = docs_dir / "pages/api" / doc_path

        if "qwip_slack" in str(module_path):
            continue
        elif module_path.name == "__init__":
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
            module = ".".join(module_path.parts[:-1])
            package = module

        elif module_path.name.startswith("_"):
            continue
        elif not (src_dir / module_path.parent / "__init__.py").exists():
            continue
        else:
            module = ".".join(module_path.parts)
            package = ".".join(module_path.parent.parts)

        if not full_doc_path.parent.exists():
            full_doc_path.parent.mkdir(parents=True)
        with open(full_doc_path, "w") as f:
            f.write(f"::: {module}")

        directory[package][module] = str(("api" / doc_path).as_posix())

    nav = list()

    for package, modules in directory.items():
        nav.append({package: [{name: file} for name, file in modules.items()]})

    return nav


def update_nav(docs_dir: Path, nav: dict):
    with open(docs_dir / "mkdocs.yml", "r") as f:
        mkdocs = yaml.safe_load(f)

    for subsection in mkdocs["nav"]:
        if "API Reference" in subsection:
            break

    subsection["API Reference"] = nav

    with open(docs_dir / "mkdocs.yml", "w") as f:
        print(f"Writing to mkdocs configuration to {docs_dir / 'mkdocs.yml'}")
        yaml.dump(
            mkdocs, f, Dumper=YAMLDumper, sort_keys=False, default_flow_style=False
        )


if __name__ == "__main__":
    src_dir = (Path(__file__).parent / "../src/").resolve()
    docs_dir = (Path(__file__).parent / "../docs").resolve()

    nav = build_api_pages(src_dir, docs_dir)
    update_nav(docs_dir, nav)
