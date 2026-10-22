"""
Grounded topic summarizer module for YouTube Lecture RAG.

Synthesizes a clean, concise technical explanation from retrieved lecture chunks
with inline clickable timestamp citations [MM:SS].

Supports:
1. Groq (if GROQ_API_KEY is set - ultra-fast 500+ tok/s)
2. Gemini (if GEMINI_API_KEY is set)
3. Fallback Grounded Transcript Synthesizer (instant, zero-API-key fallback)
"""

from __future__ import annotations

import os
import re
from typing import Optional
from .models import Chunk, RetrievalResult


SYSTEM_PROMPT = """You are a precise, technical Computer Science Teaching Assistant.
Answer the student's doubt using ONLY the provided lecture transcript excerpts.
Rules:
1. Be direct and technical. No conversational fluff ("Sure, let me explain...").
2. Cite the exact moment using inline timestamp markers like [MM:SS] where the explanation occurs.
3. Structure your response into:
   - **Direct Answer / Intuition**: 1-2 clear sentences.
   - **Key Steps**: 2-3 concise bullets with [MM:SS] citations.
   - **Complexity / Takeaway**: Time and space complexity if mentioned.
4. If the transcript does not contain the answer, say "Not covered in the provided lecture clips."
"""


class GroundedSummarizer:
    """Generates grounded explanations from retrieved lecture moments."""

    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

    def _call_groq(self, query: str, context: str) -> Optional[str]:
        if not self.groq_key:
            return None
        try:
            from groq import Groq
            client = Groq(api_key=self.groq_key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Student Doubt: {query}\n\nLecture Excerpts:\n{context}"},
                ],
                temperature=0.2,
                max_tokens=500,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"Groq API error: {e}")
            return None

    def _call_gemini(self, query: str, context: str) -> Optional[str]:
        if not self.gemini_key:
            return None
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"{SYSTEM_PROMPT}\n\nStudent Doubt: {query}\n\nLecture Excerpts:\n{context}"
            resp = model.generate_content(prompt)
            return resp.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None

    def _fallback_synthesize(self, query: str, results: list[dict]) -> str:
        """
        High-precision extractive grounded synthesis.
        Extracts salient instructional sentences and annotates with clickable timestamps.
        """
        if not results:
            return "No relevant moments found in the indexed lectures."

        top = results[0]
        ts = top["timestamp_display"]
        title = top["video_title"]
        snippet = top["snippet"]

        # Clean speech filler words
        cleaned_snippet = re.sub(r"\b(hey everyone|welcome back to the channel|so guys|um|uh)\b", "", snippet, flags=re.I)
        sentences = [s.strip() for s in re.split(r"[.!?]+", cleaned_snippet) if len(s.strip()) > 15]

        # Select top explanatory sentences
        core_sentences = sentences[:3] if len(sentences) >= 3 else sentences
        summary_body = ". ".join(core_sentences) + "." if core_sentences else snippet[:180] + "..."

        out = []
        out.append(f"**Grounded Explanation** (from *{title}* @ [{ts}]):")
        out.append(f"\n{summary_body}\n")
        out.append("**Key Lecture Moments:**")
        for r in results[:3]:
            out.append(f"- **[{r['timestamp_display']}]** in *{r['video_title']}*: \"{r['snippet'][:110]}...\"")

        return "\n".join(out)

    def summarize(self, query: str, results: list[dict]) -> str:
        """
        Synthesize answer from ranked results.
        results: list of dicts with 'video_title', 'timestamp_display', 'seek_sec', 'snippet'.
        """
        if not results:
            return "No matching content found for this query in the indexed course lectures."

        # Format context for LLM
        context_blocks = []
        for i, r in enumerate(results[:3], start=1):
            context_blocks.append(
                f"[Moment {i}] Video: {r['video_title']} @ [{r['timestamp_display']}] (seek: {r['seek_sec']}s)\n"
                f"Transcript: \"{r['snippet']}\"\n"
            )
        context = "\n".join(context_blocks)

        # 1. Try Groq (ultra fast)
        groq_resp = self._call_groq(query, context)
        if groq_resp:
            return groq_resp

        # 2. Try Gemini
        gemini_resp = self._call_gemini(query, context)
        if gemini_resp:
            return gemini_resp

        # 3. Deterministic Grounded Extractive Synthesizer
        return self._fallback_synthesize(query, results)
