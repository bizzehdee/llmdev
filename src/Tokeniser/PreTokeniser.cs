using System.Text.RegularExpressions;

namespace Tokeniser;

/// <summary>
/// Splits text into chunks (word/number/punctuation/whitespace runs) before
/// BPE training and encoding (TASK-022), so a learned merge can never join
/// across a chunk boundary in ways that produce bad tokens - e.g. merging a
/// trailing space into the next word, or digits into an arbitrarily long
/// number token. Deliberately not GPT-2's original splitting regex - it
/// under-handles Unicode letters outside ASCII and puts no cap on how many
/// digits a single "number" chunk can span - modelled instead on the
/// pattern GPT-4/<c>cl100k_base</c> tokenisers use (digit runs capped at 3,
/// contractions split out, symbol/punctuation runs kept together). The
/// reference pattern uses possessive quantifiers for pathological-input
/// performance; this uses their non-possessive equivalents for
/// portability - a correctness-neutral simplification at the scale this
/// project runs at, not a behavioural one.
/// </summary>
public static class PreTokeniser
{
    private static readonly Regex Pattern = new(
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|\s?[^\s\p{L}\p{N}]+[\r\n]*|\s+(?!\S)|\s+",
        RegexOptions.Compiled);

    /// <summary>
    /// Splits <paramref name="text"/> into chunks, in order. Every
    /// character belongs to exactly one chunk: the main alternatives'
    /// character classes - letter, number, "other" (symbol/punctuation/
    /// control), whitespace - between them partition the whole of Unicode,
    /// so nothing is silently skipped (verified by an exact encode/decode
    /// roundtrip test, not just asserted here).
    /// </summary>
    public static IEnumerable<string> Split(string text)
    {
        foreach (Match match in Pattern.Matches(text))
        {
            yield return match.Value;
        }
    }
}
