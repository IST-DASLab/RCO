"""Auto-apply the vLLM pruned-model patch when Python starts.

Drop this file alongside vllm_pruned_patch.py and put the directory on
PYTHONPATH (or sys.path), and the patch is installed before any vLLM
import. Useful for command-line invocations like:

    PYTHONPATH=loaders vllm serve <model>
    PYTHONPATH=loaders python -m lm_eval --model vllm ...

If vllm or the patch can't be imported the file silently does nothing.
"""

try:
    import vllm_pruned_patch
    vllm_pruned_patch.apply()
except Exception:
    pass
