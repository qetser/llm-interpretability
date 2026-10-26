export async function getModels() {
  const r = await fetch("/api/models");
  return r.json();
}

export async function getWorkers() {
  const r = await fetch("/api/workers");
  return r.json();
}

export async function getResults() {
  const r = await fetch("/api/results");
  return r.json();
}

// Streams tokens from /api/chat (server-sent events over POST). Resolves when the stream ends.
export async function streamChat(body, onToken, signal) {
  const r = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 2);
      if (!line.startsWith("data:")) continue;
      const ev = JSON.parse(line.slice(5));
      if (ev.error) throw new Error(ev.error);
      if (ev.token) onToken(ev.token);
    }
  }
}
