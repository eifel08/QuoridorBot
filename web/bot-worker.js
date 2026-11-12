// Runs the alpha-beta search off the main thread so the page stays responsive.
import { search } from "./bot.js";

self.onmessage = (evt) => {
  const { id, snapshot, depth, timeLimitMs, seed } = evt.data;
  const result = search(snapshot, depth, timeLimitMs, seed ?? null);
  self.postMessage({ id, ...result });
};
