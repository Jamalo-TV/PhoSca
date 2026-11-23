import { useId } from "react";

export function printSizeInputs(size) {
  return { width: size?.width?.toString() ?? "", height: size?.height?.toString() ?? "" };
}

export function parsePrintSize(inputs) {
  const widthInput = inputs.width.trim();
  const heightInput = inputs.height.trim();
  if (!widthInput && !heightInput) return { size: null, error: "" };

  const asDecimal = (input) => /^\+?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:e[+-]?\d+)?$/i.test(input)
    ? Number(input.replace(",", "."))
    : NaN;
  const width = asDecimal(widthInput);
  const height = asDecimal(heightInput);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return { size: null, error: "Gib zwei positive Masse in Millimetern ein oder lass beide Felder leer." };
  }
  if (width / height < 0.1 || width / height > 10) {
    return { size: null, error: "Das Verhältnis von Breite zu Höhe muss zwischen 1:10 und 10:1 liegen." };
  }
  return { size: { width, height }, error: "" };
}

export function PrintSizeFields({ value, onChange, disabled, error }) {
  const hintId = useId();
  const errorId = useId();
  const description = error ? `${hintId} ${errorId}` : hintId;

  return (
    <fieldset className="print-size-fields" disabled={disabled}>
      <legend>Gemessene Abzugsmasse (optional)</legend>
      <p id={hintId}>Miss die markierten Kanten am physischen Abzug. Die Masse bestimmen das Seitenverhältnis.</p>
      <div className="print-size-inputs">
        <label>
          Breite · Kante 1–2 (mm)
          <input
            type="text"
            inputMode="decimal"
            value={value.width}
            onChange={(event) => onChange({ ...value, width: event.target.value })}
            aria-describedby={description}
            aria-invalid={Boolean(error)}
          />
        </label>
        <label>
          Höhe · Kante 2–3 (mm)
          <input
            type="text"
            inputMode="decimal"
            value={value.height}
            onChange={(event) => onChange({ ...value, height: event.target.value })}
            aria-describedby={description}
            aria-invalid={Boolean(error)}
          />
        </label>
      </div>
      {error && <p className="print-size-error" id={errorId} role="status">{error}</p>}
      {(value.width || value.height) && (
        <button className="icon-button" type="button" onClick={() => onChange(printSizeInputs(null))}>
          Masse löschen
        </button>
      )}
    </fieldset>
  );
}
