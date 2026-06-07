import time
from simplemem import SimpleMem

t0 = time.perf_counter()
mem = SimpleMem()  # auto mode
print(f"[init]          {time.perf_counter() - t0:.3f}s")

t0 = time.perf_counter()
mem.add_dialogue(
    "Alice",
    "Bob, let's meet at Starbucks tomorrow at 2pm",
    "2025-11-15T14:30:00",
)
print(f"[add_dialogue 1] {time.perf_counter() - t0:.3f}s")

t0 = time.perf_counter()
mem.add_dialogue(
    "Bob",
    "Sure, I'll bring the market analysis report",
    "2025-11-15T14:31:00",
)
print(f"[add_dialogue 2] {time.perf_counter() - t0:.3f}s")

t0 = time.perf_counter()
mem.finalize()
print(f"[finalize]       {time.perf_counter() - t0:.3f}s")

t0 = time.perf_counter()
answer = mem.ask("When and where will Alice and Bob meet?")
print(f"[ask]            {time.perf_counter() - t0:.3f}s")
print(f"Answer: {answer}")
# → "16 November 2025 at 2:00 PM at Starbucks"
