#!/bin/bash
# Builds the scratch repository $WORK/repo the launchers run from: a copy of
# the code under test (af3_tpu, cloud, harness, targets, inputs) and the
# frozen inputs of the 10 pilot targets, committed so the launchers can read a
# commit. Also copies the v6e probe's 7U3J per-sample structures and
# confidences to $WORK/ref/7U3J when they exist on this machine (the
# samples-plan and stack tests compare against them; skipped otherwise).
. "$(dirname "$0")/common.sh"
set -e
R=$WORK/repo
rm -rf "$R"; mkdir -p "$R/data/inputs"
cd "$REPO"
tar cf - --exclude=__pycache__ af3_tpu cloud harness targets inputs .gitignore README.md | tar xf - -C "$R"
for t in 7U3J 7D5C 7NP6 7V3N 7BTT 7VBU 7NPL 7VC5 7XQZ 8EYE; do cp "data/inputs/$t.json" "$R/data/inputs/"; done
REF=results/af3/20261008T165218Z_v6e_probe/tpu_xla_rep1_seed1_fresh/7U3J/af3_output/7U3J
rm -rf "$WORK/ref"
if [ -d "$REF" ]; then
  for d in "$REF"/seed-1_sample-*; do
    mkdir -p "$WORK/ref/7U3J/$(basename "$d")"
    cp "$d"/*_model.cif "$d"/*confidences.json "$WORK/ref/7U3J/$(basename "$d")/"
  done
fi
cd "$R" && git init -q && git add -A && git -c user.name=test -c user.email=test@invalid commit -qm test
echo ">> scratch repository: $R$([ -d "$WORK/ref" ] && echo "; v6e reference structures: $WORK/ref/7U3J")"
