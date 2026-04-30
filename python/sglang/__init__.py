# SGLang public APIs
#
# Keep `import sglang` lightweight: defer optional frontend/runtime dependencies
# via LazyImport so that srt-only components (and unit tests) don't require
# installing the full frontend stack (e.g. IPython).

from sglang.global_config import global_config
from sglang.utils import LazyImport
from sglang.version import __version__

# Frontend Language APIs (lazy)
Runtime = LazyImport("sglang.lang.api", "Runtime")
assistant = LazyImport("sglang.lang.api", "assistant")
assistant_begin = LazyImport("sglang.lang.api", "assistant_begin")
assistant_end = LazyImport("sglang.lang.api", "assistant_end")
flush_cache = LazyImport("sglang.lang.api", "flush_cache")
function = LazyImport("sglang.lang.api", "function")
gen = LazyImport("sglang.lang.api", "gen")
gen_int = LazyImport("sglang.lang.api", "gen_int")
gen_string = LazyImport("sglang.lang.api", "gen_string")
get_server_info = LazyImport("sglang.lang.api", "get_server_info")
image = LazyImport("sglang.lang.api", "image")
select = LazyImport("sglang.lang.api", "select")
separate_reasoning = LazyImport("sglang.lang.api", "separate_reasoning")
set_default_backend = LazyImport("sglang.lang.api", "set_default_backend")
system = LazyImport("sglang.lang.api", "system")
system_begin = LazyImport("sglang.lang.api", "system_begin")
system_end = LazyImport("sglang.lang.api", "system_end")
user = LazyImport("sglang.lang.api", "user")
user_begin = LazyImport("sglang.lang.api", "user_begin")
user_end = LazyImport("sglang.lang.api", "user_end")
video = LazyImport("sglang.lang.api", "video")

RuntimeEndpoint = LazyImport("sglang.lang.backend.runtime_endpoint", "RuntimeEndpoint")
greedy_token_selection = LazyImport("sglang.lang.choices", "greedy_token_selection")
token_length_normalized = LazyImport("sglang.lang.choices", "token_length_normalized")
unconditional_likelihood_normalized = LazyImport(
    "sglang.lang.choices", "unconditional_likelihood_normalized"
)

Anthropic = LazyImport("sglang.lang.backend.anthropic", "Anthropic")
LiteLLM = LazyImport("sglang.lang.backend.litellm", "LiteLLM")
OpenAI = LazyImport("sglang.lang.backend.openai", "OpenAI")
VertexAI = LazyImport("sglang.lang.backend.vertexai", "VertexAI")

# Runtime Engine APIs
ServerArgs = LazyImport("sglang.srt.server_args", "ServerArgs")
Engine = LazyImport("sglang.srt.entrypoints.engine", "Engine")

__all__ = [
    "Engine",
    "Runtime",
    "assistant",
    "assistant_begin",
    "assistant_end",
    "flush_cache",
    "function",
    "gen",
    "gen_int",
    "gen_string",
    "get_server_info",
    "image",
    "select",
    "separate_reasoning",
    "set_default_backend",
    "system",
    "system_begin",
    "system_end",
    "user",
    "user_begin",
    "user_end",
    "video",
    "RuntimeEndpoint",
    "greedy_token_selection",
    "token_length_normalized",
    "unconditional_likelihood_normalized",
    "ServerArgs",
    "Anthropic",
    "LiteLLM",
    "OpenAI",
    "VertexAI",
    "global_config",
    "__version__",
]
