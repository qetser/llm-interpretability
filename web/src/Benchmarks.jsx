import React, { useEffect, useState } from "react";
import { getResults } from "./api.js";

const short = (id) => id.split("/").pop();
const pct = (x) => (x == null ? null : x * 100);
const fmt = (x, d = 1) => (x == null ? "–" : x.toFixed(d));

function Rate({ value, tone }) {
  if (value == null) return <span className="na">–</span>;
  return (
    <span className={`rate ${tone || ""}`}>
      <span className="bar"><i style={{ width: `${Math.max(0, Math.min(100, value))}%` }} /></span>
      <span className="num">{value.toFixed(1)}%</span>
    </span>
  );
}

function Delta({ before, after, lowerIsBetter = false }) {
  if (before == null || after == null) return <span className="na">–</span>;
  const d = after - before;
  const good = lowerIsBetter ? d < 0 : d > 0;
  const cls = Math.abs(d) < 0.05 ? "flat" : good ? "up" : "down";
  return <span className={`delta num ${cls}`}>{d > 0 ? "+" : ""}{d.toFixed(1)}</span>;
}

function Panel({ title, note, children }) {
  return (
    <section className="panel">
      <header><h2>{title}</h2>{note && <p>{note}</p>}</header>
      {children}
    </section>
  );
}

function Grouped({ groups, rows }) {
  // groups: [{label, cols:[label]}]; rows: [[name, [cell...]]]
  return (
    <div className="tbl-scroll">
      <table>
        <thead>
          <tr className="group">
            <th rowSpan={2}>Model</th>
            {groups.map((g) => <th key={g.label} colSpan={g.cols.length}>{g.label}</th>)}
          </tr>
          <tr>
            {groups.flatMap((g, gi) => g.cols.map((c, ci) => <th key={gi + c + ci} className={ci === 0 ? "sep" : ""}>{c}</th>))}
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, cells]) => (
            <tr key={name}>
              <td>{name}</td>
              {cells.map((c, i) => <td key={i} className={cells.sepAt?.has(i) ? "sep" : ""}>{c}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// marks the first cell of each group so the separator lines up with the header
function withSeps(cells, widths) {
  const sepAt = new Set();
  let i = 0;
  widths.forEach((w) => { sepAt.add(i); i += w; });
  cells.sepAt = sepAt;
  return cells;
}

const PLOTS = [["ablation_scores", "Ablating on harmful prompts"], ["actadd_scores", "Adding on harmless prompts"], ["kl_div_scores", "KL on harmless prompts"]];

export default function Benchmarks({ models }) {
  const [res, setRes] = useState(null);
  const [plotModel, setPlotModel] = useState(null);
  useEffect(() => { getResults().then((r) => { setRes(r); setPlotModel(Object.keys(r)[0] ?? null); }); }, []);

  if (!res) return <div className="bench"><div className="bench-inner"><div className="skeleton" /><div className="skeleton" /></div></div>;
  const ids = Object.keys(res);
  if (!ids.length) return <div className="empty"><h2>No benchmark results yet</h2><p>Run the pipeline for at least one model.</p></div>;

  const datasets = [...new Set(ids.flatMap((i) => Object.keys(res[i].safety || {}).filter((d) => d !== "harmless")))];
  const CONDS = [["baseline", "Baseline", ""], ["ablation", "Ablated", "warm"], ["actadd_remove", "−Steer", "warm"]];
  const safetyGroups = datasets.map((d) => ({ label: d, cols: CONDS.map((c) => c[1]) }));
  const safetyRows = ids.map((i) => [short(i), withSeps(datasets.flatMap((d) =>
    CONDS.map(([k, , tone]) => <Rate value={pct(res[i].safety?.[d]?.[k]?.refusal_rate_substring)} tone={tone} />)), datasets.map(() => 3))]);

  const hGroups = [{ label: "Harmless prompts", cols: ["Baseline refusal", "+Steer refusal", "Induced"] }];
  const hRows = ids.map((i) => {
    const b = pct(res[i].safety?.harmless?.baseline?.refusal_rate_substring);
    const s = pct(res[i].safety?.harmless?.actadd_induce?.refusal_rate_substring);
    return [short(i), [<Rate value={b} />, <Rate value={s} tone="accent" />, <Delta before={b} after={s} />]];
  });

  const tasks = [...new Set(ids.flatMap((i) => Object.keys(res[i].capability?.baseline || {})))];
  const capGroups = tasks.map((t) => ({ label: t, cols: ["Base", "Ablated", "Δ pp"] }));
  const capRows = ids.map((i) => [short(i), withSeps(tasks.flatMap((t) => {
    const b = pct(res[i].capability?.baseline?.[t]), a = pct(res[i].capability?.ablation?.[t]);
    return [<span className="num">{fmt(b)}</span>, <span className="num">{fmt(a)}</span>, <Delta before={b} after={a} />];
  }), tasks.map(() => 3))]);

  const lossGroups = [{ label: "Cross-entropy (nats)", cols: ["Baseline", "Ablated", "Δ", "−Steer", "Δ"] }, { label: "Selected direction", cols: ["Layer", "Position", "KL"] }];
  const lossRows = ids.map((i) => {
    const l = res[i].loss || {}, m = res[i].meta || {};
    return [short(i), withSeps([
      <span className="num">{fmt(l.baseline, 3)}</span>, <span className="num">{fmt(l.ablation, 3)}</span>, <Delta before={l.baseline} after={l.ablation} lowerIsBetter />,
      <span className="num">{fmt(l.actadd_remove, 3)}</span>, <Delta before={l.baseline} after={l.actadd_remove} lowerIsBetter />,
      <span className="num">{m.layer ?? "–"}</span>, <span className="num">{m.pos ?? "–"}</span>, <span className="num">{fmt(m.scores?.kl_div_score, 4)}</span>,
    ], [5, 3])];
  });

  return (
    <div className="bench">
      <div className="bench-inner">
        <div className="page-head">
          <h1>Benchmarks</h1>
          <p>How each model behaves with its refusal direction removed (ablated), subtracted at one layer (−Steer) or added (+Steer).</p>
          <div className="legend">
            <span><i className="sw" style={{ background: "var(--fg-3)" }} />Unmodified model</span>
            <span><i className="sw" style={{ background: "var(--warm)" }} />Refusal suppressed</span>
            <span><i className="sw" style={{ background: "var(--accent)" }} />Refusal induced</span>
          </div>
        </div>

        {datasets.length > 0 && (
          <Panel title="Refusal rate on harmful prompts" note="Share of responses that refuse (substring match). Low under Ablated means the intervention bypasses refusal.">
            <Grouped groups={safetyGroups} rows={safetyRows} />
          </Panel>
        )}
        <Panel title="Over-refusal and induced refusal" note="Harmless prompts. Baseline refusal is false positives; +Steer shows how far adding the direction pushes the model to refuse.">
          <Grouped groups={hGroups} rows={hRows} />
        </Panel>
        {tasks.length > 0 && (
          <Panel title="Capability" note="Accuracy under lm-evaluation-harness. Δ is Ablated minus Baseline in percentage points; green means no loss.">
            <Grouped groups={capGroups} rows={capRows} />
          </Panel>
        )}
        <Panel title="Fluency cost and selected direction" note="Cross-entropy on the model's own harmless completions. Lower is better; KL is the shift in next-token distribution when ablating.">
          <Grouped groups={lossGroups} rows={lossRows} />
        </Panel>

        <Panel title="Direction selection" note="Refusal score by source layer and token position. Each line is one position.">
          <div className="panel-tools">
            <select className="select" aria-label="Model" value={plotModel ?? ""} onChange={(e) => setPlotModel(e.target.value)}>
              {ids.map((i) => <option key={i} value={i}>{short(i)}</option>)}
            </select>
          </div>
          {plotModel && (
            <div className="plots">
              {PLOTS.map(([name, cap]) => (
                <figure key={name}>
                  <img loading="lazy" alt={cap} src={`/api/plot/${short(plotModel)}/${name}`} />
                  <figcaption>{cap}</figcaption>
                </figure>
              ))}
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
