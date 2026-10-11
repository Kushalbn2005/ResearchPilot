def complete(system: str, user: str, max_tokens: int = 4000, json_mode: bool = False) -> str:
    """The only place that talks to an LLM (Groq).

    gpt-oss models "think" before answering and that thinking counts against max_tokens,
    so we keep reasoning effort low and the token budget generous. If Groq's strict JSON
    mode rejects a generation, we retry once without it; qa.parse_and_validate() still
    guarantees the final shape.
    """
    from groq import BadRequestError, Groq

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

    if json_mode:
        try:
            # Groq's JSON mode requires the word "JSON" in the prompt (qa.SYSTEM has it).
            resp = client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
            return resp.choices[0].message.content or ""
        except BadRequestError:
            pass  # fall through to a plain completion
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""