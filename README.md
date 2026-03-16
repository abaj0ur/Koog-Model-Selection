# Koog — Automatic Model Selection: API Design Proposal

Conceptual API extension for the Koog framework that removes the manual model-selection bottleneck.

## The problem

Every `execute()` call today requires a concrete `LLModel`. Picking the right one means reading provider docs, comparing costs, checking capabilities — every time.

## The proposal

One new `execute()` overload that accepts a lambda instead of a model:

```kotlin
// before
executor.execute(prompt, tools, OpenAIModels.Chat.GPT4o)

// after
executor.execute(prompt, tools) {
    withCapability(LLMCapability.ToolCalling)
    maxCostPerInputMillionTokens(20.0)
    fastest()
}
```

Existing call sites compile unchanged.

## File

`ModelOptimizerAPI.kt` — full API reference with types, examples, pros/cons, and implementation notes.

> Conceptual design only — not a runnable implementation.
