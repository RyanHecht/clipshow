// Account control for the top-right of a page, and a sign-in dialog (Jellyfin password or Quick Connect).
// Usage: Signin.account(element, authState); Signin.open() to ask for sign-in from anywhere (e.g. "Open in Jellyfin").
(() => {
  const css = `
  .si-dialog { border: 1px solid var(--line, #d3dde6); border-radius: 16px; padding: 22px; width: min(92vw, 380px);
               background: var(--paper, #fff); color: var(--ink, #1e2833); font: 16px/1.45 "Atkinson Hyperlegible", system-ui, sans-serif; }
  .si-dialog::backdrop { background: rgba(0,0,0,.45); }
  .si-dialog h2 { font: 600 22px/1.2 Fredoka, system-ui, sans-serif; margin: 0 0 4px; }
  .si-dialog p { margin: 0 0 12px; color: var(--muted, #5c6b78); font-size: 14px; }
  .si-dialog label { display: block; font-weight: 700; font-size: 14px; margin: 10px 0 4px; }
  .si-dialog input { width: 100%; box-sizing: border-box; padding: 9px 11px; font: inherit; color: inherit; background: var(--sky, #e9eff4); border: 1px solid var(--line, #d3dde6); border-radius: 8px; }
  .si-dialog button { font: inherit; font-weight: 700; cursor: pointer; border-radius: 8px; padding: 9px 12px; }
  .si-primary { width: 100%; margin-top: 14px; border: 0; background: var(--sweater, var(--accent, #f2b632)); color: var(--sweater-ink, var(--accent-ink, #3a2a00)); }
  .si-secondary { width: 100%; border: 1px solid var(--line, #d3dde6); background: transparent; color: inherit; }
  .si-or { display: flex; align-items: center; gap: 10px; color: var(--muted, #5c6b78); font-size: 13px; margin: 14px 0; }
  .si-or::before, .si-or::after { content: ""; flex: 1; border-top: 1px solid var(--line, #d3dde6); }
  .si-code { font: 600 36px/1 Fredoka, monospace; letter-spacing: .15em; text-align: center; margin: 8px 0; }
  .si-err { color: var(--danger, #b3261e); font-weight: 700; min-height: 1.3em; margin: 10px 0 0 !important; }
  .si-close { position: absolute; top: 10px; right: 12px; border: 0 !important; background: none; color: var(--muted, #5c6b78); font-size: 20px; padding: 4px 8px !important; }
  .si-account { display: flex; gap: 12px; align-items: center; font-size: 14px; color: var(--muted, #5c6b78); white-space: nowrap; }
  .jf-logo { width: 1.15em; height: 1.15em; vertical-align: -0.2em; flex: none; }
  .si-who { display: inline-flex; align-items: center; gap: 6px; color: var(--ink, #1e2833); font-weight: 700; }
  .si-account button.si-jfbtn { display: inline-flex; align-items: center; gap: 7px; text-decoration: none; padding: 5px 11px; border-radius: 999px;
                                border: 1px solid var(--line, #d3dde6); background: var(--paper, #fff); }
  .si-account button.si-jfbtn:hover { border-color: #00A4DC; }
  .si-dialog .si-brand { display: block; width: 44px; height: 44px; margin-bottom: 10px; }
  .si-dialog .si-note { font-size: 13px; margin: 14px 0 0; }
  .si-account a, .si-account button { color: var(--ink, #1e2833); font: inherit; font-weight: 700; background: none; border: 0; padding: 0; cursor: pointer; text-decoration: underline; text-underline-offset: 3px; }
  .si-dialog [hidden] { display: none !important; }`;
  document.head.append(Object.assign(document.createElement("style"), { textContent: css }));

  let state = null, dialog = null;

  async function post(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `Sign-in failed (HTTP ${r.status}).`);
    return data;
  }

  function build() {
    dialog = document.createElement("dialog");
    dialog.className = "si-dialog";
    dialog.innerHTML = `
      <button class="si-close" aria-label="Close">×</button>
      <img src="/static/jellyfin.svg" alt="Jellyfin" class="si-brand">
      <h2>Sign in with Jellyfin</h2>
      <p class="si-lead">Use your Jellyfin account. Signing in lets you open moments in Jellyfin.</p>
      <form class="si-pw">
        <label>Jellyfin username <input name="u" autocomplete="username" required></label>
        <label>Jellyfin password <input name="p" type="password" autocomplete="current-password"></label>
        <button class="si-primary">Sign in with Jellyfin</button>
      </form>
      <div class="si-qc" hidden>
        <div class="si-or">or</div>
        <button class="si-secondary si-qc-start" type="button">Use Quick Connect</button>
        <div class="si-qc-wait" hidden><div class="si-code" aria-live="polite"></div>
          <p style="text-align:center">In a Jellyfin app you're signed in to, open Quick Connect and enter this code.</p></div>
      </div>
      <p class="si-err" role="alert"></p>
      <p class="si-note">Clip Show checks your password with your Jellyfin server and doesn't keep it.</p>`;
    document.body.append(dialog);
    const q = s => dialog.querySelector(s);
    const err = q(".si-err");
    const done = () => location.reload();
    q(".si-close").addEventListener("click", () => dialog.close());
    q(".si-pw").addEventListener("submit", async e => {
      e.preventDefault(); err.textContent = "";
      try { await post("/api/auth/login", { username: q('[name="u"]').value, password: q('[name="p"]').value }); done(); }
      catch (ex) { err.textContent = ex.message; }
    });
    q(".si-qc-start").addEventListener("click", async () => {
      err.textContent = "";
      try {
        const { code, secret } = await post("/api/auth/quickconnect");
        q(".si-code").textContent = code; q(".si-qc-wait").hidden = false; q(".si-qc-start").hidden = true;
        const poll = async () => {
          if (!dialog.open) return;
          try { const r = await post("/api/auth/quickconnect/check", { secret }); if (r.pending) setTimeout(poll, 2000); else done(); }
          catch (ex) { err.textContent = ex.message; q(".si-qc-wait").hidden = true; q(".si-qc-start").hidden = false; }
        };
        setTimeout(poll, 2000);
      } catch (ex) { err.textContent = ex.message; }
    });
  }

  function open(reason) {
    if (!dialog) build();
    dialog.querySelector(".si-qc").hidden = !(state && state.quick_connect);
    dialog.querySelector(".si-lead").textContent = reason || "Use your Jellyfin account. Signing in lets you open moments in Jellyfin.";
    dialog.querySelector(".si-err").textContent = "";
    dialog.showModal();
    dialog.querySelector('[name="u"]').focus();
  }

  function account(box, s) {
    state = s;
    box.classList.add("si-account");
    box.replaceChildren();
    const el = (tag, props) => Object.assign(document.createElement(tag), props);
    if (s.user) {
      const who = el("span", { className: "si-who", title: `Signed in with Jellyfin as ${s.user.name}` });
      who.innerHTML = '<img src="/static/jellyfin.svg" alt="Jellyfin:" class="jf-logo">';
      who.append(s.user.name);
      box.append(who);
      if (s.user.admin) box.append(el("a", { href: "/admin", textContent: "Manage" }));
      const out = el("button", { textContent: "Sign out" });
      out.addEventListener("click", async () => { await fetch("/api/auth/logout", { method: "POST" }); location.reload(); });
      box.append(out);
    } else {
      const b = el("button", { className: "si-jfbtn" });
      b.innerHTML = '<img src="/static/jellyfin.svg" alt="" class="jf-logo">';
      b.append("Sign in with Jellyfin");
      b.addEventListener("click", () => open());
      box.append(b);
    }
  }

  window.Signin = { open, account, signedIn: () => !!(state && state.user) };
})();
