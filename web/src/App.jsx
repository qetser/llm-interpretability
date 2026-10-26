import React, { useEffect, useState } from "react";
import { BarChart3, MessageSquare, Microscope } from "lucide-react";
import { getModels, getWorkers } from "./api.js";
import Chat from "./Chat.jsx";
import Benchmarks from "./Benchmarks.jsx";

function useMobile() {
  const q = "(max-width: 700px)";
  const [m, setM] = useState(() => window.matchMedia(q).matches);
  useEffect(() => {
    const mq = window.matchMedia(q);
    const on = () => setM(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return m;
}

export default function App() {
  const mobile = useMobile();
  const [models, setModels] = useState([]);
  const [selected, setSelected] = useState(null);
  const [workers, setWorkers] = useState([]);
  const [tab, setTab] = useState(() => (window.location.hash === "#bench" ? "bench" : "chat"));
  useEffect(() => { window.history.replaceState(null, "", tab === "bench" ? "#bench" : "#"); }, [tab]);

  useEffect(() => {
    getModels().then((ms) => {
      setModels(ms);
      setSelected((s) => s ?? ms.find((m) => m.probed)?.id ?? null);
    });
  }, []);

  useEffect(() => {
    let alive = true;
    const tick = () => getWorkers().then((w) => alive && setWorkers(w)).catch(() => {});
    tick();
    const id = setInterval(tick, 5000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const model = models.find((m) => m.id === selected);
  if (mobile) {
    // minimal phone flow: one chat screen, no rail, no benchmarks
    return (
      <div className="app mobile">
        <main className="main">
          {model ? (
            <Chat key={model.id} model={model} mobile models={models} onSelect={setSelected} />
          ) : (
            <div className="empty"><h2>No probed models yet</h2></div>
          )}
        </main>
      </div>
    );
  }
  return (
    <div className="app">
      <aside className="rail">
        <div className="brand">
          <span className="mark"><Microscope size={16} /></span>
          Refusal Probe
        </div>
        <nav className="nav">
          <div className="segmented" role="tablist">
            <button role="tab" aria-selected={tab === "chat"} onClick={() => setTab("chat")}><MessageSquare size={15} />Chat</button>
            <button role="tab" aria-selected={tab === "bench"} onClick={() => setTab("bench")}><BarChart3 size={15} />Benchmarks</button>
          </div>
        </nav>
        <div className="section-label">Models</div>
        <div className="models">
          {models.map((m) => (
            <button
              key={m.id}
              className="model"
              disabled={!m.probed}
              aria-current={m.id === selected}
              title={m.probed ? m.id : "Run the pipeline for this model first"}
              onClick={() => { setSelected(m.id); setTab("chat"); }}
            >
              <span className={`dot ${m.loaded ? "live" : m.probed ? "on" : ""}`} />
              <span className="name">{m.id.split("/").pop()}</span>
              <span className="sub">{m.probed ? `Direction at layer ${m.layer}${m.loaded ? " · on GPU" : ""}` : "Not probed yet"}</span>
            </button>
          ))}
        </div>
        {workers.length > 0 && (
          <>
            <div className="section-label">GPUs</div>
            <ul className="gpus">
              {workers.map((w, i) => (
                <li key={w.url} title={w.gpu}>
                  <span className={`dot ${!w.up ? "" : w.busy || w.queued ? "live" : "on"}`} />
                  <span>GPU {i}</span>
                  <span className="gsub">{!w.up ? "down" : `${w.free_gb.toFixed(1)} GB free${w.loaded ? ` · ${w.loaded}` : ""}`}</span>
                </li>
              ))}
            </ul>
          </>
        )}
        <div className="rail-foot">For studying refusal behaviour in open-weight models.</div>
      </aside>
      <main className="main">
        {tab === "bench" ? (
          <Benchmarks models={models} />
        ) : model ? (
          <Chat key={model.id} model={model} />
        ) : (
          <div className="empty">
            <h2>No probed models yet</h2>
            <p>Run <code>scripts/run_all.sh</code> to extract a refusal direction, then reload.</p>
          </div>
        )}
      </main>
    </div>
  );
}
