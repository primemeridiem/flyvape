#!/usr/bin/env bash
# Fetch the FlyEM male CNS v1.0 connectome (CC-BY, HHMI Janelia FlyEM, Cambridge
# Connectomics Group, Google Research) into data/, under the names build_graph.py reads.
# The README's URLs dropped the flat-connectome/ path segment and 404; these are the live ones.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
BASE=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
get() {  # $1 remote name, $2 local name
  if [ -s "data/$2" ]; then echo "have data/$2"; return; fi
  echo "fetching $1 -> data/$2"
  curl -fL --retry 3 -o "data/$2.part" "$BASE/$1" && mv "data/$2.part" "data/$2"
}
get body-annotations-male-cns-v1.0-minconf-0.5.feather body-annotations.feather
get body-neurotransmitters-male-cns-v1.0.feather body-neurotransmitters.feather
# traced-only (508 MB) is enough: build_graph.py keeps only Traced bodies anyway.
# FLY_FULL_WEIGHTS=1 fetches the full 1.05 GB table instead.
if [ "${FLY_FULL_WEIGHTS:-0}" = "1" ]; then
  get connectome-weights-male-cns-v1.0-minconf-0.5.feather connectome-weights.feather
else
  get connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather connectome-weights.feather
fi
echo done
