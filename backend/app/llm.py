import time


def complete(system: str, user: str, max_tokens: int = 4000, json_mode: bool = False) -> str:
    """The only place that talks to an LLM (Groq).

    gpt-oss models "think" before answering and that thinking counts against max_tokens,
    so we keep reasoning effort low and the token budget generous. If Groq's strict JSON
    mode rejects a generation, we retry once without it; qa.parse_and_validate() still
    guarantees the final shape. Rate-limit errors (free-tier tokens per minute) are
    retried with a short wait.
    """
    from groq import BadRequestError, Groq, RateLimitError

    from .config import settings

    client = Groq(api_key=settings.llm_api_key)
    kwargs = dict(
        model=settings.llm_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    if settings.llm_model.startswith("openai/gpt-oss"):
        kwargs["extra_body"] = {"reasoning_effort": "low"}

    def create(**kw):
        for attempt in range(3):
            try:
                return client.chat.completions.create(**kw)
            except RateLimitError:
                if attempt == 2:
                    raise
                time.sleep(8 * (attempt + 1))  # 8 s, then 16 s

    if json_mode:
        try:
            # Groq's JSON mode requires the word "JSON" in the prompt (our prompts have it).
            resp = create(**kwargs, response_format={"type": "json_object"})
            return resp.choices[0].message.content or ""
        except BadRequestError:
            pass  # fall through to a plain completion
    resp = create(**kwargs)
    return resp.choices[0].message.content or ""