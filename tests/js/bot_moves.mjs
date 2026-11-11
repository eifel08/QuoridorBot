// Reads {states, depths} JSON on stdin and prints the JS bot's choice for
// every (state, depth) pair (used by tests/test_bot.py).
import { search } from "../../web/bot.js";

let input = "";
process.stdin.on("data", (d) => (input += d));
process.stdin.on("end", () => {
  const { states, depths } = JSON.parse(input);
  const out = [];
  for (const s of states) for (const d of depths) out.push(search(s, d, null));
  process.stdout.write(JSON.stringify(out));
});
