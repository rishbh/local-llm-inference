"""Every setting, read from the environment.

Nothing secret is written in this file. The access code comes from the
environment (see .env.example), so it never reaches git and never reaches the
browser.

The sampling settings deliberately match `local-medgemma.py`'s defaults. If the
website answers with different settings than the benchmark, the benchmark stops
predicting what users will see.
"""
import os


def _load_dotenv():
    """Read ./.env into the environment, if it exists.

    Hand-rolled rather than adding python-dotenv for twelve lines. A real
    environment variable always wins, so `ACCESS_CODE=x python server.py`
    overrides the file.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def _str(name, default=""):
    return os.environ.get(name, default).strip()


def _int(name, default):
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _float(name, default):
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _bool(name, default=False):
    v = _str(name).lower()
    return default if not v else v in ("1", "true", "yes", "on")


# ---- Where the model lives -------------------------------------------------
OLLAMA_URL = _str("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL = _str("MODEL", "medgemma1.5")

# ---- Generation: keep these equal to the benchmark's defaults --------------
NUM_CTX = _int("NUM_CTX", 4096)
MAX_TOKENS = _int("MAX_TOKENS", 1600)
TEMPERATURE = _float("TEMPERATURE", 0.0)
REPEAT_PENALTY = _float("REPEAT_PENALTY", 1.1)
REPEAT_LAST_N = _int("REPEAT_LAST_N", 256)

# ---- Protecting a machine that can only answer one question at a time ------
# The benchmark peaked at 16.78 GB of 16.9 GB with a single request in flight.
# There is no room for a second one, so answers are serialised rather than
# allowed to fight over memory and swap.
MAX_CONCURRENT = _int("MAX_CONCURRENT", 1)
MAX_QUEUE = _int("MAX_QUEUE", 8)

# Two of the benchmark's 50 questions took 657 s and 844 s when the machine
# started swapping. Without a ceiling, one of those blocks the queue for
# everybody behind it.
ANSWER_TIMEOUT_S = _float("ANSWER_TIMEOUT_S", 120.0)
FIRST_TOKEN_TIMEOUT_S = _float("FIRST_TOKEN_TIMEOUT_S", 45.0)

# ---- Who is allowed in -----------------------------------------------------
# Empty means no gate at all. The server prints a loud warning when that is the
# case, because the tunnel URL is public to anyone who learns it.
ACCESS_CODE = _str("ACCESS_CODE")
RATE_LIMIT_QUESTIONS = _int("RATE_LIMIT_QUESTIONS", 10)
RATE_LIMIT_WINDOW_S = _float("RATE_LIMIT_WINDOW_S", 600.0)

MAX_QUESTION_CHARS = _int("MAX_QUESTION_CHARS", 500)

# ---- Images ----------------------------------------------------------------
# Only takes effect if the model actually has a vision encoder: /api/health
# asks Ollama for the model's capabilities and the page hides the button when
# "vision" is absent, so turning this on cannot make a text-only model pretend.
ALLOW_IMAGES = _bool("ALLOW_IMAGES", True)

# 4 MB of decoded image. The page shrinks anything bigger to 1024px before
# uploading, so this ceiling is really a guard against a crafted request, not
# against an ordinary phone photo.
MAX_IMAGE_BYTES = _int("MAX_IMAGE_BYTES", 4 * 1024 * 1024)

# Only these three. Each is checked by its magic bytes, not by what the browser
# claims the type is.
ALLOWED_IMAGE_TYPES = ("png", "jpeg", "webp")

# Both timeouts are multiplied by this when an image is attached. Pushing a
# picture through the vision encoder on a CPU happens *before* the first token
# appears, so a 45 s first-token budget that is generous for text will kill an
# image question that was going to work.
IMAGE_TIMEOUT_FACTOR = _float("IMAGE_TIMEOUT_FACTOR", 3.0)

# ---- Logging ---------------------------------------------------------------
# OFF by default, and it must stay that way unless you have decided what you
# are doing about consent and retention. A public health Q&A page collects
# real symptoms from real people: that is sensitive personal data under the
# DPDP rules, not test data. With this off, the log keeps timings and a salted
# hash of the caller, never the question text.
LOG_QUESTIONS = _bool("LOG_QUESTIONS", False)
LOG_PATH = _str("LOG_PATH", os.path.join(os.path.dirname(__file__), "logs", "requests.jsonl"))

HOST = _str("HOST", "127.0.0.1")
PORT = _int("PORT", 8000)


# ---- The system prompt -----------------------------------------------------
# Copied verbatim from DEFAULT_SYSTEM in local-medgemma.py. test_server.py
# fails if the two ever drift apart, so the site and the benchmark cannot
# quietly start measuring different things.
SYSTEM_PROMPT = (
    "You are a medical information assistant answering questions from patients. "
    "Begin your reply with the answer itself. Do not write out your reasoning, "
    "a plan, or numbered thinking steps first. "
    "Answer in plain language a non-medical person can follow. Be accurate and "
    "specific. Put any red-flag symptoms that need urgent care, and when the "
    "person should see a doctor, near the start rather than the end. Say "
    "\"I don't know\" rather than guessing a gene name, a number, or a drug "
    "name you are not sure of. Do not diagnose, and do not give a dose or a "
    "prescription. If the question is unclear or unsafe to answer without an "
    "examination, say so."
)

# Appended to SYSTEM_PROMPT *only* when an image is attached. Kept separate on
# purpose: SYSTEM_PROMPT itself must stay byte-identical to the benchmark's
# DEFAULT_SYSTEM, and the benchmark has never sent an image. A text-only
# question therefore still gets exactly the prompt that was measured.
IMAGE_NOTE = (
    " An image is attached to this question. Describe only what you can "
    "actually see in it. Do not name a diagnosis from an image. If the image "
    "is blurred, cropped, or is not a medical image at all, say that instead "
    "of guessing, and say that an image alone is not enough to judge this."
)
