import React, { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { Check, Copy } from "lucide-react";

export function CopyButton({ getText, label = "Copy", className = "" }) {
  const [done, setDone] = useState(false);
  async function copy() {
    const text = getText();
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // clipboard API needs HTTPS/localhost; fall back for plain-http access
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    setDone(true);
    setTimeout(() => setDone(false), 1600);
  }
  return (
    <button type="button" className={`copy ${className}`} onClick={copy} aria-label={done ? "Copied" : label}>
      {done ? <Check size={14} /> : <Copy size={14} />}
      <span>{done ? "Copied" : label}</span>
    </button>
  );
}

function CodeBlock({ children }) {
  const ref = useRef(null);
  const child = React.Children.toArray(children)[0];
  const lang = /language-([\w-]+)/.exec(child?.props?.className || "")?.[1];
  return (
    <div className="codeblock">
      <div className="codehead">
        <span className="lang">{lang || "text"}</span>
        <CopyButton getText={() => ref.current?.innerText.replace(/\n$/, "") ?? ""} />
      </div>
      <pre ref={ref}>{children}</pre>
    </div>
  );
}

const components = {
  pre: CodeBlock,
  a: ({ node, ...p }) => <a {...p} target="_blank" rel="noreferrer noopener" />,
  table: ({ node, ...p }) => <div className="md-table"><table {...p} /></div>,
};

export default function Markdown({ children }) {
  return (
    <div className="md">
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]} components={components}>
        {children}
      </ReactMarkdown>
    </div>
  );
}
