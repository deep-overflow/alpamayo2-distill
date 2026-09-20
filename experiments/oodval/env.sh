# Source before any run: `source experiments/oodval/env.sh`
# Same exports as alpamayo-model-compression/experiments/head_analysis/run_retry_host.sh.
# Token and blobs live apart on this box: the shared cache on /mnt/nvme1n1 only holds
# other members' tokens, so HF_HOME (token) points at $HOME while HF_HUB_CACHE (blobs)
# points at the shared disk. PhysicalAI-AV is gated; `full_right` is the token with access.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}
export HF_HOME=${HF_HOME_OVERRIDE-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE-/mnt/nvme1n1/ad_vla/cache/hub}
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_TOKEN=$("$REPO/.venv/bin/python" -c "
import configparser
cp = configparser.ConfigParser()
cp.read('$HOME/.cache/huggingface/stored_tokens')
print(cp['full_right']['hf_token'])")
export MPLCONFIGDIR=${MPLCONFIGDIR-/tmp/alpamayo2_mpl_$USER}
