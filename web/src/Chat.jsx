import React, { useEffect, useRef, useState } from "react";
import { ArrowUp, ChevronLeft, Eraser, Square } from "lucide-react";
import { streamChat } from "./api.js";
import Markdown, { CopyButton } from "./Markdown.jsx";

const MODES = [
  { id: "baseline", label: "Baseline", hint: "Unmodified model." },
  { id: "ablate", label: "Ablate refusal", hint: "Projects the refusal direction out of every layer.", warm: true },
  { id: "steer", label: "Steer", hint: "Adds the refusal direction at one layer. Positive induces refusal, negative suppresses it.", warm: true },
];
const STARTERS = ["Explain how photosynthesis works.", "How do locks get picked?", "Write a short poem about autumn."];

function Transcript({ messages, busy, onStarter }) {
  const end = useRef(null);
  useEffect(() => end.current?.scrollIntoView({ block: "end" }), [messages]);
  return (
    <div className="transcript">
      <div className="thread">
        {messages.length === 0 && (
          <div className="empty">
            <h2>Start a conversation</h2>
            <p className="desktop-only">Switch modes in the toolbar to compare how the same prompt behaves with and without the refusal direction.</p>
            {onStarter && (
              <div className="starters">
                {STARTERS.map((s) => <button key={s} className="btn" onClick={() => onStarter(s)}>{s}</button>)}
              </div>
            )}
          </div>
        )}
        {messages.map((m, i) => {
          const streaming = busy && i === messages.length - 1 && m.role === "assistant";
          return (
            <div key={i} className={`msg ${m.role}`}>
              <span className="who">{m.role === "user" ? "You" : "Model"}</span>
              {m.role === "assistant" ? (
                <>
                  <div className={`body ${streaming ? "caret" : ""}`}>{m.content ? <Markdown>{m.content}</Markdown> : null}</div>
                  {!streaming && m.content && <CopyButton className="msg-copy" label="Copy" getText={() => m.content} />}
                </>
              ) : (
                <div className="body">{m.content}</div>
              )}
            </div>
          );
        })}
        <div ref={end} />
      </div>
    </div>
  );
}

function MobileSetup({ model, models, onSelect, system, setSystem, onStart }) {
  const probed = models.filter((m) => m.probed);
  return (
    <div className="setup">
      <div className="setup-body">
        <h1>Refusal ablated</h1>
        {probed.length > 1 ? (
          <select className="select" aria-label="Model" value={model.id} onChange={(e) => onSelect(e.target.value)}>
            {probed.map((m) => <option key={m.id} value={m.id}>{m.id.split("/").pop()}</option>)}
          </select>
        ) : (
          <p className="mono">{model.id.split("/").pop()}</p>
        )}
        <label htmlFor="sys">System prompt</label>
        <textarea id="sys" rows={6} placeholder="Optional. Leave empty for the model default." value={system} onChange={(e) => setSystem(e.target.value)} />
      </div>
      <button className="btn primary start" onClick={onStart}>Start chat</button>
    </div>
  );
}

export default function Chat({ model, mobile = false, models = [], onSelect }) {
  const [started, setStarted] = useState(!mobile);
  const [mode, setMode] = useState("ablate");
  const [strength, setStrength] = useState(1);
  const [compare, setCompare] = useState(false);
  const [temperature, setTemperature] = useState(0);
  const [system, setSystem] = useState("");
  const [panes, setPanes] = useState({ main: [], base: [] });
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const abort = useRef(null);
  const ta = useRef(null);

  useEffect(() => {
    const el = ta.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [input]);

  const patch = (key, fn) => setPanes((p) => ({ ...p, [key]: fn(p[key]) }));

  async function run(key, paneMode, history, text) {
    patch(key, (m) => [...m, { role: "user", content: text }, { role: "assistant", content: "" }]);
    await streamChat(
      { model: model.id, messages: [...history, { role: "user", content: text }], mode: paneMode, strength, temperature, system: system || null },
      (tok) => patch(key, (m) => [...m.slice(0, -1), { ...m[m.length - 1], content: m[m.length - 1].content + tok }]),
      abort.current.signal
    );
  }

  async function send(override) {
    const text = (override ?? input).trim();
    if (!text || busy) return;
    setInput("");
    setError(null);
    setBusy(true);
    abort.current = new AbortController();
    try {
      // the server serialises generation, so the panes run one after the other
      if (compare) await run("base", "baseline", panes.base, text);
      await run("main", mode, panes.main, text);
    } catch (e) {
      if (e.name !== "AbortError") setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  if (mobile && !started) {
    return <MobileSetup model={model} models={models} onSelect={onSelect} system={system} setSystem={setSystem} onStart={() => setStarted(true)} />;
  }
  if (mobile) {
    // fixed to ablate-refusal at temperature 0; the system prompt was set on the previous screen
    return (
      <div className="chat">
        <div className="m-head">
          <button className="btn quiet icon" aria-label="Edit system prompt" disabled={busy} onClick={() => setStarted(false)}><ChevronLeft size={18} /></button>
          <span className="m-title">{model.id.split("/").pop()}</span>
          <span className="chip warm">ablated</span>
          <button className="btn quiet icon" aria-label="Clear conversation" disabled={busy} onClick={() => setPanes({ main: [], base: [] })}><Eraser size={16} /></button>
        </div>
        <div className="panes">
          <section className="pane"><Transcript messages={panes.main} busy={busy} /></section>
        </div>
        <div className="composer-wrap">
          {error && <div className="error" role="alert">{error}</div>}
          <div className="composer">
            <textarea ref={ta} rows={1} value={input} aria-label="Message" placeholder="Message" enterKeyHint="send" onChange={(e) => setInput(e.target.value)} />
            {busy ? (
              <button className="btn send" aria-label="Stop generating" onClick={() => abort.current?.abort()}><Square size={14} fill="currentColor" /></button>
            ) : (
              <button className="btn primary send" aria-label="Send message" disabled={!input.trim()} onClick={() => send()}><ArrowUp size={18} /></button>
            )}
          </div>
        </div>
      </div>
    );
  }

  const cur = MODES.find((m) => m.id === mode);
  const steerLabel = `${strength > 0 ? "+" : ""}${strength.toFixed(2)}`;
  return (
    <div className="chat">
      <div className="toolbar">
        <div className="title">
          <h2>{model.id.split("/").pop()}</h2>
          {model.layer != null && <span className="chip mono">layer {model.layer}/{model.n_layers}</span>}
        </div>
        <div className="segmented" role="group" aria-label="Mode">
          {MODES.map((m) => (
            <button key={m.id} className={m.warm ? "warm" : ""} aria-pressed={mode === m.id} title={m.hint} onClick={() => setMode(m.id)}>{m.label}</button>
          ))}
        </div>
        {mode === "steer" && (
          <label className="field">Strength
            <input type="range" min="-2" max="3" step="0.25" value={strength} onChange={(e) => setStrength(+e.target.value)} />
            <span className="val num">{steerLabel}</span>
          </label>
        )}
        <label className="field">Temperature
          <input type="range" min="0" max="1.5" step="0.1" value={temperature} onChange={(e) => setTemperature(+e.target.value)} />
          <span className="val num">{temperature.toFixed(1)}</span>
        </label>
        <label className="switch">
          <input type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} />
          <span className="track" />Compare
        </label>
        <button className="btn quiet icon" title="Clear conversation" aria-label="Clear conversation" onClick={() => setPanes({ main: [], base: [] })} disabled={busy}><Eraser size={16} /></button>
      </div>
      <div className="subbar">
        <span>{cur.hint}</span>
        <input aria-label="System prompt" placeholder="System prompt (optional)" value={system} onChange={(e) => setSystem(e.target.value)} />
      </div>

      <div className={`panes ${compare ? "two" : ""}`}>
        {compare && (
          <section className="pane">
            <div className="pane-head"><span className="chip">Baseline</span></div>
            <Transcript messages={panes.base} busy={busy} />
          </section>
        )}
        <section className="pane">
          {compare && <div className="pane-head"><span className={`chip ${cur.warm ? "warm" : ""}`}>{cur.label}{mode === "steer" ? ` ${steerLabel}` : ""}</span></div>}
          <Transcript messages={panes.main} busy={busy} onStarter={compare ? undefined : (s) => send(s)} />
        </section>
      </div>

      <div className="composer-wrap">
        {error && <div className="error" role="alert">{error}</div>}
        <div className="composer">
          <textarea
            ref={ta}
            rows={1}
            value={input}
            aria-label="Message"
            placeholder={`Message ${model.id.split("/").pop()}`}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
            }}
          />
          {busy ? (
            <button className="btn send" aria-label="Stop generating" onClick={() => abort.current?.abort()}><Square size={14} fill="currentColor" /></button>
          ) : (
            <button className="btn primary send" aria-label="Send message" disabled={!input.trim()} onClick={() => send()}><ArrowUp size={18} /></button>
          )}
        </div>
        <div className="hint-foot">Enter to send · Shift+Enter for a new line</div>
      </div>
    </div>
  );
}
