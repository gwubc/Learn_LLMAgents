import os
import re

from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def search(query: str) -> str:
    result = tavily.search(query=query, max_results=3)
    snippets = [r["content"] for r in result.get("results", [])]
    return "\n\n".join(snippets) if snippets else "No results found."


TOOLS = {
    "search": {
        "fn": search,
        "description": (
            "search[query] — Search the web for current information. "
        ),
    }
}

TOOL_DESCRIPTIONS = "\n".join(
    f"  - {info['description']}" for info in TOOLS.values()
)

SYSTEM_PROMPT = f"""You are a research assistant
Available tools:
{TOOL_DESCRIPTIONS}

Response format (follow strictly):
  Thought: <your reasoning about what to do next>
  Action: <tool_name>[<input>]

When you have enough information to answer:
  Thought: <final reasoning>
  Action: Finish[<your complete answer>]

Rules:
- Every response MUST contain exactly one Thought: line followed by exactly one Action: line.
- Never write a plain answer — always end with either a tool Action or Finish[answer].
- Use search to look up facts you are uncertain about.
- Call Finish[answer] as soon as you have enough information to give a complete answer.
"""

ACTION_RE = re.compile(r"Action:\s*(\w+)\[(.+?)\]", re.DOTALL)


def parse_action(text: str) -> tuple[str, str] | None:
    match = ACTION_RE.search(text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None


class ReActAgent:
    def __init__(self, max_steps: int = 10):
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_URL"),
        )
        self.model = os.getenv("LLM_MODEL")
        self.max_steps = max_steps
        print(f"[INIT] Model : {self.model}")
        print(f"[INIT] Base URL: {os.getenv('LLM_URL')}")

    def run(self, question: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        print(f"\n{'='*60}")
        print(f"[QUESTION] {question}")
        print(f"{'='*60}")

        for step in range(1, self.max_steps + 1):
            print(f"\n[STEP {step}] Calling LLM...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            msg = response.choices[0].message
            llm_text = msg.content or ""
            print(f"[STEP {step}] finish_reason={response.choices[0].finish_reason}")
            print(f"[STEP {step}] LLM output:\n{llm_text}")

            # Append the assistant turn
            messages.append({"role": "assistant", "content": llm_text})

            parsed = parse_action(llm_text)
            if parsed is None:
                print("[WARN] Could not parse Action — stopping.")
                return llm_text

            action_name, action_input = parsed

            if action_name == "Finish":
                print(f"\n{'='*60}")
                print("[FINAL ANSWER]")
                print(action_input)
                print(f"{'='*60}")
                return action_input

            if action_name not in TOOLS:
                observation = f"Error: unknown tool '{action_name}'."
            else:
                print(f"[TOOL CALL] {action_name}({action_input!r})")
                observation = TOOLS[action_name]["fn"](action_input)
                print(f"[OBSERVATION]\n{observation}")

            messages.append({
                "role": "user",
                "content": f"Observation: {observation}",
            })

        return "Max steps reached without a final answer."


if __name__ == "__main__":
    agent = ReActAgent(max_steps=8)
    agent.run( "Who is the current CEO of the company that acquired GitHub, and what is their net worth?")


"""
[INIT] Model : gemma-4-31b-it
[INIT] Base URL: https://generativelanguage.googleapis.com/v1beta/openai/

============================================================
[QUESTION] Who is the current CEO of the company that acquired GitHub, and what is their net worth?
============================================================

[STEP 1] Calling LLM...
[STEP 1] finish_reason=stop
[STEP 1] LLM output:
<thought>The user is asking for the current CEO of the company that acquired GitHub and that CEO's net worth.

1.  **Identify the company that acquired GitHub.** (GitHub was acquired by Microsoft).
2.  **Identify the current CEO of that company.** (Microsoft's CEO is Satya Nadella).
3.  **Find the current net worth of that CEO.** (Search for Satya Nadella's net worth).

Plan:
1. Search for "company that acquired GitHub".
2. Search for "CEO of [Company Name]".
3. Search for "[CEO Name] net worth".</thought>Thought: GitHub was acquired by Microsoft. I need to verify the current CEO of Microsoft and their current net worth.
Action: search[current CEO of Microsoft and their net worth]
[TOOL CALL] search('current CEO of Microsoft and their net worth')
[OBSERVATION]
Satya Nadella, Microsoft CEO, earned $96.5 million. What's his net worth?

Satya Nadella. CEO, Microsoft. $1.3B. $3M (0.23%). Real Time Net Worth. as of 5/28/26. Reflects change since 5 pm ET of prior trading day. #2877 in the world

According to Forbes World's Billionaires List, he is the 5th richest person with a real-time net worth of USD 131 billion. He is the 30th

[STEP 2] Calling LLM...
[STEP 2] finish_reason=stop
[STEP 2] LLM output:
<thought>The search results provide conflicting information.
- One snippet says "Satya Nadella. CEO, Microsoft. $1.3B."
- Another snippet mentions "According to Forbes World's Billionaires List, he is the 5th richest person with a real-time net worth of USD 131 billion." This seems highly unlikely for Satya Nadella, as $131 billion would place him among the richest people in the world (like Bill Gates or Jeff Bezos), which is not the case for Nadella. The $1.3B figure is more plausible for a high-earning CEO who is not the founder.

Wait, looking closely at the second snippet: "According to Forbes World's Billionaires List, he is the 5th richest person with a real-time net worth of USD 131 billion." This sounds like it might be referring to someone else or is a hallucination/error in the snippet text.

Let's verify Satya Nadella's net worth using a more reliable source or checking for discrepancies. Usually, Nadella's net worth is cited in the billions, but not in the hundreds of billions.

Let's double-check the $1.3B vs $131B.
Actually, looking at the first snippet: "$1.3B" is a common figure cited for him.
The second snippet says "5th richest person... USD 131 billion". This is definitely wrong for Satya Nadella.

I will confirm the net worth one more time to be precise.</thought>Thought: The search results provided conflicting information, with one suggesting a net worth of $1.3 billion and another claiming $131 billion (which is likely an error or referring to another person). I will perform a targeted search to confirm Satya Nadella's current estimated net worth.
Action: search[Satya Nadella net worth 2024 2025]
[TOOL CALL] search('Satya Nadella net worth 2024 2025')
[OBSERVATION]
* Satya Nadella net worth includes salary, bonuses, and..., Microsoft boss per year earns Rs..., is he richer than Googles Sundar Pichai? # Satya Nadella net worth includes salary, bonuses, and…, Microsoft boss per year earns Rs…, is he richer than Google’s Sundar Pichai? ## Satya Nadella who is a technology executive and business leader earns primarily from his position as CEO of Microsoft, as well as from board memberships, book royalties, and investments. https://www.india.com/business/satya-nadella-net-worth-includes-salary-bonuses-and-bonds-microsoft-boss-per-year-earning-is-he-richer-than-googles-sundar-pichai-7878420/. In the year 2024, Satya Nadella will get a total salary and allowance of $79.1 million (Rs 670 crores), which is 63 percent more than the year 2023. **Satya Nadella has an estimated net worth of $500 million in 2025 whereas Google’s CEO, Sundar Pichai’s Wealth in 2025 has been calculated at Rs 110 crore.** Satya Nadella who is a technology executive and business leader earns primarily from his position as CEO of Microsoft, as well as from board memberships, book royalties, and investments.

# Satya Nadella is netting a record $96.5 million pay package as Microsoft's AI bets paid off. Satya Nadella's record $96.5 million payday shows how Microsoft's AI boom is rewarding its top leader. Satya Nadella is getting his biggest payday since becoming Microsoft's CEO a decade ago. The chief executive's pay package jumped to a record $96.5 million for fiscal 2025, about a 22% increase from $79 million in 2024, according to the company's proxy filing on Tuesday. The bulk of Nadella's pay — about $84.2 million — came from stock awards tied to Microsoft's market value during the AI boom. His total compensation has risen over the last decade: about $18 million in 2015, $55 million in 2022, and $79 million in 2024, before reaching this year's high of $96.5 million. Chong Ming is a junior reporter at Business Insider's Singapore bureau, covering the latest in AI and tech. Look out for an alert in your inbox the next time a new story is published!

What is Satya Nadella's net worth in 2025? As of October 2024, Nadella's net worth had risen to around $1.4 billion, according to Benzinga

[STEP 3] Calling LLM...
[STEP 3] finish_reason=stop
[STEP 3] LLM output:
<thought>The company that acquired GitHub is Microsoft.
The current CEO of Microsoft is Satya Nadella.
Regarding net worth:
- One source says estimated net worth of $500 million in 2025 (this seems low compared to other sources).
- Benzinga (as of October 2024) says around $1.4 billion.
- Another source mentioned $1.3B in the first search.
- The $131 billion figure from the first search was clearly an error/hallucination in the snippet.

The most consistent and plausible figures for his overall net worth (rather than annual salary) are around $1.3 billion to $1.4 billion. Annual compensation is cited around $96.5 million for fiscal 2025.

I have enough information to answer.</thought>Thought: GitHub was acquired by Microsoft, and the current CEO of Microsoft is Satya Nadella. Based on recent reports, Satya Nadella's estimated net worth is approximately $1.3 billion to $1.4 billion.
Action: Finish[The company that acquired GitHub is **Microsoft**, and its current CEO is **Satya Nadella**. His estimated net worth is approximately **$1.3 billion to $1.4 billion**.]

============================================================
[FINAL ANSWER]
The company that acquired GitHub is **Microsoft**, and its current CEO is **Satya Nadella**. His estimated net worth is approximately **$1.3 billion to $1.4 billion**.
============================================================
"""