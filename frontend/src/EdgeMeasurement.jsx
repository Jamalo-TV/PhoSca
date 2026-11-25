import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { measurePageEdge } from "./api.js";
import { getSourceEdgeTraces } from "./CropVariants.jsx";

const labels = ["Anfang", "Zwischenpunkt", "Ende"];
const errors = {
  edge_trace_points_too_close: "Markiere einen längeren Abschnitt der Kante.",
  edge_trace_invalid_middle: "Setze den Zwischenpunkt auf die Kante zwischen Anfang und Ende.",
  edge_trace_image_too_large: "Die Originalseite ist für diese Messung zu gross.",
  edge_trace_source_unavailable: "Die Originalseite kann nicht geöffnet werden.",
};

function clamp(value) {
  return Math.min(1, Math.max(0, value));
}

export function MeasuredEdgeOverlay({ traces, size }) {
  return traces.map((trace, index) => (
    <polyline key={index} className="measured-edge-line" vectorEffect="non-scaling-stroke" fill="none"
      points={trace.map(({ x, y }) => size
        ? `${1000 * (x * (size.width - 1) + .5) / size.width},${1000 * (y * (size.height - 1) + .5) / size.height}`
        : `${1000 * x},${1000 * y}`).join(" ")} />
  ));
}

export default function EdgeMeasurement({ pageId, onClose, onMeasured }) {
  const heading = useId();
  const dialog = useRef(null);
  const viewport = useRef(null);
  const drawing = useRef(null);
  const pendingFocus = useRef(null);
  const request = useRef(null);
  const [size, setSize] = useState(null);
  const [zoom, setZoom] = useState(false);
  const [points, setPoints] = useState([]);
  const [selected, setSelected] = useState(0);
  const [traces, setTraces] = useState([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    dialog.current.showModal();
    return () => request.current?.abort();
  }, []);

  function centerAt(point) {
    if (!point || !viewport.current || !drawing.current) return;
    const frame = viewport.current;
    frame.scrollLeft = point.x * drawing.current.clientWidth - frame.clientWidth / 2;
    frame.scrollTop = point.y * drawing.current.clientHeight - frame.clientHeight / 2;
  }

  useLayoutEffect(() => {
    if (pendingFocus.current) {
      centerAt(pendingFocus.current);
      pendingFocus.current = null;
    }
  }, [zoom]);

  function select(index) {
    setSelected(index);
    if (zoom) centerAt(points[index]);
  }

  function updatePoint(point) {
    const updated = [...points];
    updated[selected] = point;
    setPoints(updated);
    setTraces([]);
    setMessage("");
    onMeasured([]);
  }

  function place(event) {
    if (!size || busy) return;
    const rect = drawing.current.getBoundingClientRect();
    updatePoint({
      x: clamp(((event.clientX - rect.left) * size.width / rect.width - .5) / (size.width - 1)),
      y: clamp(((event.clientY - rect.top) * size.height / rect.height - .5) / (size.height - 1)),
    });
    if (selected === points.length && selected < 2) setSelected(selected + 1);
  }

  function moveSelected(delta) {
    if (!points[selected] || busy) return;
    updatePoint({
      x: clamp(points[selected].x + delta[0] / (size.width - 1)),
      y: clamp(points[selected].y + delta[1] / (size.height - 1)),
    });
  }

  function nudge(event) {
    const increments = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (points[selected] && increments[event.key]) {
      event.preventDefault();
      moveSelected(increments[event.key]);
    }
  }

  async function measure() {
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setMessage("");
    try {
      const result = await measurePageEdge(pageId, points, controller.signal);
      if (controller.signal.aborted || !dialog.current?.open) return;
      const found = getSourceEdgeTraces(result.traces);
      setTraces(found);
      onMeasured(found);
      setMessage(found.length ? "Prüfe am Original, ob die gestrichelten Abschnitte zur Papierkante gehören."
        : "Hier war keine eindeutige Kante messbar. Vergrössere die Originalpixel und setze die Punkte genauer auf die sichtbare Kante.");
    } catch (error) {
      if (!controller.signal.aborted) setMessage(errors[error.message] || "Die Kante konnte nicht gemessen werden.");
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  }

  return (
    <dialog ref={dialog} aria-labelledby={heading} className="edge-measurement-dialog"
      onCancel={() => request.current?.abort()} onClose={onClose}>
      <header className="edge-measurement-header">
        <h2 id={heading}>Kante am Original messen</h2>
        <button type="button" onClick={() => {
          request.current?.abort();
          dialog.current.close();
        }}>Schliessen</button>
      </header>
      <p>Setze Anfang, Zwischenpunkt und Ende auf dieselbe sichtbare Kante.</p>
      <div className="edge-measurement-controls" role="group" aria-label="Messpunkte">
        {labels.map((label, index) => (
          <button type="button" key={label} aria-pressed={selected === index}
            disabled={busy || index > points.length} onClick={() => select(index)}>
            {index + 1} {label}
          </button>
        ))}
        <button type="button" disabled={!size} onClick={() => {
          pendingFocus.current = points[selected] || points.at(-1) || { x: .5, y: .5 };
          setZoom(!zoom);
        }}>
          {zoom ? "Bildübersicht" : "Originalpixel vergrössern"}
        </button>
      </div>
      {zoom && points[selected] && (
        <div className="edge-measurement-controls" role="group" aria-label="Messpunkt um einen Originalpixel verschieben">
          {[["←", "Links", [-1, 0]], ["↑", "Oben", [0, -1]], ["↓", "Unten", [0, 1]], ["→", "Rechts", [1, 0]]]
            .map(([symbol, direction, delta]) => <button type="button" key={direction} aria-label={direction}
              disabled={busy} onClick={() => moveSelected(delta)}>{symbol}</button>)}
        </div>
      )}
      <div className="edge-measurement-viewport" ref={viewport} tabIndex={0} onKeyDown={nudge}
        aria-label="Messfläche. Pfeiltasten verschieben den gewählten Punkt um einen Originalpixel.">
        <div className="edge-measurement-image" ref={drawing}
          style={size ? { "--edge-image-aspect": size.width / size.height, width: zoom ? size.width * 2 : undefined } : undefined}
          onClick={place}>
          <img src={`/api/v1/pages/${pageId}/image`} alt="Originalseite für die Kantenmessung" draggable={false}
            onLoad={(event) => {
              const { naturalWidth: width, naturalHeight: height } = event.currentTarget;
              if (width > 1 && height > 1) setSize({ width, height });
              else setMessage("Die Originalseite ist zu klein für eine Messung.");
            }}
            onError={() => setMessage("Die Originalseite kann nicht geöffnet werden.")} />
          {size && <svg className="edge-measurement-overlay" aria-hidden="true" viewBox="0 0 1000 1000"
            preserveAspectRatio="none"><MeasuredEdgeOverlay traces={traces} size={size} /></svg>}
          {size && points.map((point, index) => (
            <span key={index} aria-hidden="true" className={selected === index ? "edge-anchor active" : "edge-anchor"}
              style={{ left: `${100 * (point.x * (size.width - 1) + .5) / size.width}%`,
                top: `${100 * (point.y * (size.height - 1) + .5) / size.height}%` }}><b>{index + 1}</b></span>
          ))}
        </div>
      </div>
      <footer className="edge-measurement-controls">
        <button type="button" disabled={points.length !== 3 || busy} onClick={measure}>
          {busy ? "Kante wird gemessen …" : "Kante messen"}
        </button>
        <button type="button" disabled={!points.length || busy} onClick={() => {
          setPoints([]);
          setSelected(0);
          setTraces([]);
          setMessage("");
          onMeasured([]);
        }}>Messpunkte löschen</button>
      </footer>
      {message && <p role="status">{message}</p>}
    </dialog>
  );
}
