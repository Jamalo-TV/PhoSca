import { useEffect, useState } from "react";
import App from "./App.jsx";
import { getBrowserSession, signIn, signOut } from "./api.js";

const anonymousSession = { required: false, tenant_id: null };
const privateSession = { required: true, tenant_id: null };
const anonymousStorageKey = "phosca-anonymous-deployment";

export default function BrowserSession() {
  const [session, setSession] = useState(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  async function check() {
    setMessage("");
    try {
      const status = await getBrowserSession();
      if (status.required) sessionStorage.removeItem(anonymousStorageKey);
      else sessionStorage.setItem(anonymousStorageKey, "1");
      setSession(status);
    } catch {
      if (!navigator.onLine && sessionStorage.getItem(anonymousStorageKey)) setSession(anonymousSession);
      else setMessage("PhoSca ist gerade nicht erreichbar.");
    }
  }

  useEffect(() => {
    check();
    const expire = () => {
      sessionStorage.removeItem(anonymousStorageKey);
      setSession(privateSession);
      setMessage("Bitte melde dich an.");
    };
    window.addEventListener("phosca-session-expired", expire);
    return () => window.removeEventListener("phosca-session-expired", expire);
  }, []);

  async function submit(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const apiKey = new FormData(form).get("apiKey")?.trim();
    if (!apiKey) return;
    setMessage("");
    setBusy(true);
    try {
      setSession(await signIn(apiKey));
      form.reset();
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function closeSession() {
    setBusy(true);
    setMessage("");
    try {
      await signOut();
      setSession(privateSession);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  if (session && (!session.required || session.tenant_id)) {
    return (
      <>
        {session.required && (
          <div className="browser-session-controls">
            {message && <span role="alert">{message}</span>}
            <button type="button" onClick={closeSession} disabled={busy}>Abmelden</button>
          </div>
        )}
        <App key={session.tenant_id || "anonymous"} uploadQueueScope={session.tenant_id} />
      </>
    );
  }

  return (
    <main className="browser-sign-in">
      <h1>PhoSca</h1>
      {message && <p role="alert">{message}</p>}
      {session ? (
        <form onSubmit={submit}>
          <label>API-Schlüssel
            <input type="password" name="apiKey" autoComplete="off" required maxLength={256} />
          </label>
          <button type="submit" disabled={busy}>Anmelden</button>
        </form>
      ) : message ? (
        <button type="button" onClick={check}>Erneut versuchen</button>
      ) : <p role="status">Verbindung wird geprüft …</p>}
    </main>
  );
}
