"""Pyinvoke tasks.py file for automating releases and admin stuff."""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
from pprint import pprint

import requests
from invoke import task
from monty.os import cd

with open("pyproject.toml") as f:
    for line in f:
        if line.startswith("version"):
            NEW_VER = line.split("=")[-1].strip().strip('"')
            break


@task
def make_tutorials(ctx):
    ctx.run("rm -rf docs/tutorials")
    ctx.run("jupyter nbconvert examples/*.ipynb --to=markdown --output-dir=docs/tutorials")
    for fn in glob.glob("docs/tutorials/*/*.png"):
        ctx.run(f'mv "{fn}" docs/assets')

    for fn in os.listdir("docs/tutorials"):
        lines = [
            "---",
            "layout: default",
            "title: " + fn,
            "nav_exclude: true",
            "---",
            "",
        ]
        path = f"docs/tutorials/{fn}"
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif fn.endswith(".md"):
            with open(path) as file:
                for line in file:
                    line = line.rstrip()  # noqa: PLW2901
                    if line.startswith("![png]"):
                        t1, t2 = line.split("(")
                        t2, t3 = t2.split("/")
                        lines.append(t1 + "(assets/" + t3)
                    else:
                        lines.append(line)
            with open(path, "w") as file:
                file.write("\n".join(lines))


@task
def make_docs(ctx):
    """
    This new version requires markdown builder.

        pip install sphinx-markdown-builder

    Adding the following to conf.py

        extensions = [
            'sphinx_markdown_builder'
        ]

    Build markdown files with sphinx-build command

        sphinx-build -M markdown ./ build
    """
    make_tutorials(ctx)

    with cd("docs"):
        ctx.run("cp ../README.md index.md", warn=True)
        ctx.run("rm matcalc.*.rst", warn=True)
        ctx.run("sphinx-apidoc -P -M -d 6 -o . -f ../src/matcalc")
        ctx.run("cp modules.rst index.rst")
        ctx.run("sphinx-build -M markdown . .")
        ctx.run("rm *.rst", warn=True)
        ctx.run("cp markdown/matcalc*.md .")
        for fn in glob.glob("matcalc*.md"):
            with open(fn) as f:
                lines = [line.rstrip() for line in f if "Submodules" not in line]
            if fn == "matcalc.md":
                preamble = [
                    "---",
                    "layout: default",
                    "title: API Documentation",
                    "nav_order: 5",
                    "---",
                    "",
                ]
            else:
                preamble = [
                    "---",
                    "layout: default",
                    "title: " + fn,
                    "nav_exclude: true",
                    "---",
                    "",
                ]
            with open(fn, "w") as f:
                f.write("\n".join(preamble + lines))

        ctx.run("rm -r markdown", warn=True)
        ctx.run("cp ../*.md .")
        ctx.run("mv README.md index.md")
        ctx.run("rm -rf *.orig doctrees", warn=True)

        with open("index.md") as f:
            contents = f.read()
        with open("index.md", "w") as f:
            contents = re.sub(
                r"\n## Official Documentation[^#]*",
                "{: .no_toc }\n\n## Table of contents\n{: .no_toc .text-delta }\n* TOC\n{:toc}\n\n",
                contents,
            )
            contents = "---\nlayout: default\ntitle: Home\nnav_order: 1\n---\n\n" + contents

            f.write(contents)


@task
def publish(ctx):
    ctx.run("rm dist/*.*", warn=True)
    ctx.run("python -m build")
    ctx.run("python -m build --wheel")
    ctx.run("twine upload dist/*")


@task
def release_github(ctx):  # noqa: ARG001
    desc = get_changelog()

    payload = {
        "tag_name": "v" + NEW_VER,
        "target_commitish": "main",
        "name": "v" + NEW_VER,
        "body": desc,
        "draft": False,
        "prerelease": False,
    }
    response = requests.post(
        "https://api.github.com/repos/materialsvirtuallab/matcalc/releases",
        data=json.dumps(payload),
        headers={"Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]},
        timeout=10,
    )
    pprint(response.json())


@task
def release(ctx, notest: bool = False) -> None:  # noqa: FBT001, FBT002
    ctx.run("rm -r dist build matcalc.egg-info", warn=True)
    if not notest:
        ctx.run("pytest tests")
    release_github(ctx)


def get_changelog():
    with open("changes.md") as f:
        contents = f.read()
        match = re.search(f"## v{NEW_VER}([^#]*)", contents)
        if not match:
            raise ValueError("could not parse latest version from changes.md")
        return match.group(1).strip()


@task
def view_docs(ctx) -> None:
    with cd("docs"):
        ctx.run("bundle exec jekyll serve")
