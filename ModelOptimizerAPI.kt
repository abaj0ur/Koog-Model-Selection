/**
 * Koog — Automatic Model Selection: API Design Proposal
 *
 * Core idea: one new execute() overload that takes a lambda instead of a
 * concrete LLModel. All filtering and ranking happen inside MultiLLMPromptExecutor.
 * Existing call sites compile unchanged.
 *
 * Conceptual design — not a runnable implementation.
 */

// --- The new execute overload (the only thing most users need to know) ---

interface PromptExecutor {

    suspend fun execute(
        prompt: Prompt,
        tools: List<Tool<*, *>> = emptyList(),
        model: LLModel,
    ): LLMResponse

    // note: default throws — keeps SingleLLMPromptExecutor and others valid
    //       without forcing them to implement selection logic
    suspend fun execute(
        prompt: Prompt,
        tools: List<Tool<*, *>> = emptyList(),
        selection: ModelRequirementsBuilder.() -> Unit,
    ): LLMResponse = throw UnsupportedOperationException(
        "This executor does not support requirements-based model selection."
    )

    // note: separate overload for pre-built requirements (reuse across executors)
    suspend fun execute(
        prompt: Prompt,
        tools: List<Tool<*, *>> = emptyList(),
        requirements: ModelRequirements,
    ): LLMResponse
}

// --- usage: all five task use-cases ---

executor.execute(prompt) { cheapest() }

executor.execute(prompt) { fastest() }

executor.execute(prompt) { largestContext() }

executor.execute(prompt, tools) {
    withCapability(LLMCapability.ToolCalling, LLMCapability.Vision, LLMCapability.Moderation)
    cheapest()
}

executor.execute(prompt, tools) {
    withCapability(LLMCapability.ToolCalling)
    maxCostPerInputMillionTokens(20.0)  // $0.02/token = $20/1M tokens
    minContextWindow(500_000)
    fastest()
}

// reuse across multiple executors without re-declaring the lambda
val corporatePolicy = modelRequirements {
    withCapability(LLMCapability.ToolCalling)
    maxCostPerInputMillionTokens(20.0)
    fastest()
}
executor.execute(prompt, requirements = corporatePolicy)


// --- ModelRequirementsBuilder (the DSL the lambda runs inside) ---

// Users discover everything through IDE autocomplete inside the lambda.
// No need to read docs — just open the block and see the options.


// @DslMarker prevents builder methods from leaking into enclosing scope
@DslMarker
annotation class KoogModelDsl

@KoogModelDsl
class ModelRequirementsBuilder {

    // filters run before scorer — excluded candidates never reach ranking
    fun withCapability(vararg capabilities: LLMCapability)
    fun minContextWindow(tokens: Int)
    fun maxCostPerInputMillionTokens(usd: Double)
    fun maxCostPerOutputMillionTokens(usd: Double)

    fun cheapest()
    fun fastest()
    fun largestContext()

    // custom scorer; higher = better
    // e.g. optimizeBy { it.contextWindow / (it.inputCostPer1MTokens ?: 1.0) }
    fun optimizeBy(scorer: (LLModel) -> Double)
}

/** Returns a reusable snapshot of the builder state. */
fun modelRequirements(block: ModelRequirementsBuilder.() -> Unit): ModelRequirements

data class ModelRequirements(
    val requiredCapabilities: Set<LLMCapability>,
    val minContextWindow: Int,
    val maxCostPerInputMillionTokens: Double?,
    val maxCostPerOutputMillionTokens: Double?,
    // cheapest/fastest/largestContext all compile down to a scorer so
    // DefaultModelSelector has a single ranking path for every goal
    val scorer: (LLModel) -> Double,
)


// --- ModelSelector (pluggable strategy, wired into MultiLLMPromptExecutor) ---

// Most users never touch this; exists so power users can swap the entire
// Selection algorithm (A/B testing, budget enforcement, circuit breaking, etc.)

fun interface ModelSelector {
    // null → executor throws NoModelMatchException
    fun select(candidates: List<LLModel>, requirements: ModelRequirements): LLModel?
}

class MultiLLMPromptExecutor(
    vararg providers: Pair<LLMProvider, LLMClient>,
    val selector: ModelSelector = DefaultModelSelector,
    val onModelSelected: ((LLModel, ModelRequirements) -> Unit)? = null,
) : PromptExecutor

object DefaultModelSelector : ModelSelector {
    override fun select(candidates: List<LLModel>, requirements: ModelRequirements): LLModel? {
        val filtered = candidates
            .filter { it.capabilities.containsAll(requirements.requiredCapabilities) }
            .filter { it.contextWindow >= requirements.minContextWindow }
            .filter { requirements.maxCostPerInputMillionTokens
                ?.let { max -> (it.inputCostPer1MTokens ?: Double.MAX_VALUE) <= max } ?: true }
            .filter { requirements.maxCostPerOutputMillionTokens
                ?.let { max -> (it.outputCostPer1MTokens ?: Double.MAX_VALUE) <= max } ?: true }

        // negated scorer lets maxByOrNull handle cheapest/fastest uniformly
        return filtered.maxByOrNull { requirements.scorer(it) }
    }
}

// --- custom selector examples ---

// A/B test: 10% of traffic to an experimental model
val abSelector = ModelSelector { candidates, requirements ->
    if (Random.nextFloat() < 0.1f)
        candidates.find { "experimental" in it.id }
            ?: DefaultModelSelector.select(candidates, requirements)
    else
        DefaultModelSelector.select(candidates, requirements)
}

// fallback: relax cost ceiling if strict requirements match nothing
val fallbackSelector = ModelSelector { candidates, requirements ->
    DefaultModelSelector.select(candidates, requirements)
        ?: DefaultModelSelector.select(candidates, requirements.copy(
            maxCostPerInputMillionTokens = null
        ))
}


// --- Changes to LLModel (two new optional fields, nothing breaks) ---

data class LLModel(
    val id: String,
    val provider: LLMProvider,
    val capabilities: List<LLMCapability> = emptyList(),
    val contextWindow: Int = 0,
    val maxOutputTokens: Int? = null,

    // null = unknown cost; treated as MAX_VALUE in DefaultModelSelector —
    // excluded from cheapest() unless the entire candidate pool is unknown
    val inputCostPer1MTokens: Double? = null,
    val outputCostPer1MTokens: Double? = null,

    // ordinal used directly by fastest(); see CONS for limitations
    val speedTier: SpeedTier? = null,
)

// enum not sealed — ordered label, not an extension point
// warn: converting to sealed interface later is binary-incompatible (breaks exhaustive when)
// users needing finer ordering should use optimizeBy { scorer }
enum class SpeedTier { Fast, Standard, Slow }

// all official model constants (OpenAIModels, AnthropicModels, etc.) need a
// one-time backfill of inputCostPer1MTokens, outputCostPer1MTokens, speedTier
// override any constant without subclassing: OpenAIModels.Chat.GPT4o.copy(inputCostPer1MTokens = 2.5)


// --- LLMCapability: v1 keeps the enum, future path to sealed interface ---
//
// Left as enum in this proposal — ToolCalling / Vision / Moderation all fit.
//
// Future path to user-defined capabilities:
//
//   sealed interface LLMCapability {
//       data object ToolCalling : LLMCapability
//       data object Vision      : LLMCapability
//       data object Moderation  : LLMCapability
//       data class Custom(val name: String) : LLMCapability
//   }
//
// warn: binary-incompatible — exhaustive when-expressions break on upgrade
// migration: @Deprecated enum typealias for one release cycle, then remove


// --- Execution flow inside execute(selection: ...) ---
//
// 1. lambda runs against ModelRequirementsBuilder → ModelRequirements
// 2. executor.models() returns LLModels from registered clients only —
//    natural allow-list: unregistered providers are never reachable
// 3. ModelSelector.select(candidates, requirements) → LLModel?
// 4. model found → delegate to existing execute(prompt, tools, model)
// 5. null → NoModelMatchException

class NoModelMatchException(
    val requirements: ModelRequirements,
    val candidates: List<LLModel>,
) : IllegalStateException(
    buildString {
        appendLine("No model satisfies the given requirements.")
        appendLine("Requirements: $requirements")
        appendLine("Candidates checked (${candidates.size}):")
        candidates.forEach { m ->
            // internal describeWhy() explains per-constraint failure:
            // "missing capabilities [Moderation]"
            // "contextWindow 128000 < required 500000"
            // "inputCost $30/1M > max $20/1M"
            appendLine("  - ${m.provider}/${m.id}: ${describeWhy(m, requirements)}")
        }
    }
)


// --- PROS ---
//
// Zero breaking changes — all existing execute(prompt, tools, model) sites
//   compile and run identically
//
// Near-zero learning curve — "lambda instead of model"; IDE autocomplete
//   inside the block surfaces every option without reading docs
//
// Clean filter / optimize separation — withCapability / minContextWindow /
//   maxCost narrow the pool; cheapest / fastest / optimizeBy rank survivors;
//   no hidden precedence rules
//
// All five task use-cases covered in 1–2 lines (Part 1)
//
// Extensible without forking Koog:
//   custom criteria   → optimizeBy { scorer }
//   custom algorithm  → ModelSelector fun interface
//   custom capability → LLMCapability.Custom (Part 4a, future)
//
// Transparent — onModelSelected for metrics; NoModelMatchException with
//   per-candidate failure reasons


// --- CONS ---
//
// Cost metadata requires ongoing maintenance — pricing changes frequently;
//   stale values make cheapest() misleading between releases
//   phase 2: load from bundled JSON config to decouple from release cycle
//
// fastest() is a heuristic — SpeedTier is a static label, not measured latency;
//   wrong for edge cases (load spikes, regional variation, prompt length)
//   phase 2: rolling TTFT averages via onModelSelected + OpenTelemetry
//
// No provider filter in DSL — "only Anthropic" requires a custom ModelSelector;
//   a future withProvider(...) method would cover this cleanly
//
// No built-in fallback chain — intentional; silent degradation is worse than
//   a loud failure; three-line custom ModelSelector handles fallback (Part 3)
//
// LLMCapability not user-extensible — enum until sealed interface migration;
//   custom capabilities require a Koog source change (Part 4a)


// --- UNDER-THE-HOOD CHANGES ---
//
// LLModel              — 3 new nullable fields; default null; no existing code breaks
// SpeedTier            — new enum, ~5 lines
// ModelRequirementsBuilder + ModelRequirements — new, ~50 lines
// modelRequirements()  — top-level fun, ~3 lines
// ModelSelector + DefaultModelSelector — new, ~30 lines
// MultiLLMPromptExecutor — 2 new constructor params + new execute() overloads, ~25 lines
// PromptExecutor       — 2 new default overloads; all existing implementations unchanged
// NoModelMatchException — new, ~15 lines + internal describeWhy()
// model constants      — one-time cost/speedTier backfill across all providers