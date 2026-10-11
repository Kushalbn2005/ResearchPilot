def complete(system: str, user: str, max_tokens: int = 1000, json_mode: bool = False) -> str:
    """The only place that talks to an LLM (Groq).

    The SDK is imported lazily so ingestion and tests need neither the SDK nor an API key.
    """
    from groq import Groq

    from .config import settings

    client = Groq(api_key=settings.llm_api_key)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,  # deterministic: easier to evaluate and debug
        max_tokens=max_tokens,
        # Groq's JSON mode requires the word "JSON" in the prompt (qa.SYSTEM has it).
        response_format={"type": "json_object"} if json_mode else None,
    )
    return resp.choices[0].message.content or ""