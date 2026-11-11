// Reads a JSON list of Python BoardState dicts on stdin and prints the JS
// engine's snapshot for each position (used by tests/test_web_engine.py).
import { fromSnapshot, snapshot } from "../../web/engine.js";

let input = "";
process.stdin.on("data", (d) => (input += d));
process.stdin.on("end", () => {
  const states = JSON.parse(input);
  process.stdout.write(JSON.stringify(states.map((s) => snapshot(fromSnapshot(s)))));
});
