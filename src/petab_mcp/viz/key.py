"""A plot's header: its title, an ⓘ that says on hover how to read the plot, and its legend, each
entry explained on hover.

Every plot has one, so the way to ask what a plot shows is the same everywhere, and no
explanation takes room until it is asked for. The texts are written for a biologist who has not
fitted a model: what is on the axes, what a good and a bad picture look like, what to do next.
"""

from __future__ import annotations

import html
import re

import panel as pn
import param

from . import style


def _html(text: str) -> str:
    """``text``, paragraphs apart by a blank line and ``**bold**``, as HTML."""
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    return "".join(
        "<p>" + re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(" ".join(p.split()))) + "</p>"
        for p in paragraphs
    )


def entry(
    label: str,
    about: str = "",
    color: str = style.MUTED,
    mark: str = "●",
    toggle: str | None = None,
) -> dict:
    """A legend entry: ``mark`` in ``color``, ``label``, and on hover ``about``. With ``toggle``,
    a checkbox before it, which adds ``toggle`` to the key's ``hidden`` when unticked."""
    return {
        "label": label,
        "about": _html(about) if about else "",
        "color": color,
        "mark": mark,
        "toggle": toggle,
    }


class Key(pn.custom.JSComponent):
    """The header; see the module. The tooltip is the page's own element, fixed to the window,
    so no plot or card it sits over clips it."""

    title = param.String(default="")
    about = param.String(default="", doc="HTML: how to read the plot.")
    entries = param.List(default=[], doc="Dicts of label, about (HTML), color, mark and toggle.")
    hidden = param.List(default=[], doc="The toggles of the entries unticked.")

    _stylesheets = [
        f"""
    .key {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 14px;
            font-size: 13px; margin: 6px 0 2px 0; }}
    .key .title {{ font-weight: 600; font-size: 14px; }}
    .key .info {{ color: {style.PALETTE[0]}; cursor: help; white-space: nowrap; }}
    .key .entry {{ white-space: nowrap; }}
    .key .entry.tip {{ cursor: help; border-bottom: 1px dotted {style.MUTED}; }}
    .key .mark {{ margin-right: 4px; }}
    .key input {{ margin: 0 4px 0 0; vertical-align: -1px; cursor: pointer; }}
    .key label.entry {{ cursor: pointer; }}
    .key .off {{ opacity: .45; }}
    """
    ]

    _esm = """
    let tip = null;
    function hide() { if (tip) { tip.remove(); tip = null; } }
    function show(target, content) {
      hide();
      tip = document.createElement("div");
      tip.innerHTML = content;
      Object.assign(tip.style, {
        position: "fixed", zIndex: 10000, maxWidth: "400px", padding: "8px 12px",
        background: "rgba(33,33,33,.96)", color: "#fff", borderRadius: "6px",
        font: "13px/1.45 system-ui, -apple-system, sans-serif", pointerEvents: "none",
        boxShadow: "0 2px 12px rgba(0,0,0,.35)" });
      tip.querySelectorAll("p").forEach((p, i, all) => {
        p.style.margin = i === all.length - 1 ? "0" : "0 0 7px 0"; });
      document.body.append(tip);
      const r = target.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
      const left = Math.max(8, Math.min(r.left, window.innerWidth - w - 8));
      const below = r.bottom + 6;
      tip.style.left = left + "px";
      tip.style.top = (below + h > window.innerHeight - 8 ? Math.max(8, r.top - h - 6) : below) + "px";
    }
    export function render({ model, el }) {
      const root = document.createElement("div"); root.className = "key";
      function item(cls, about, tag = "span") {
        const s = document.createElement(tag); s.className = cls;
        if (about) {
          s.classList.add("tip");
          s.addEventListener("mouseenter", () => show(s, about));
          s.addEventListener("mouseleave", hide);
        }
        return s;
      }
      function draw() {
        root.innerHTML = "";
        if (model.title) {
          const t = item("title", ""); t.textContent = model.title; root.append(t);
        }
        if (model.about) {
          const i = item("info", model.about); i.textContent = "ⓘ how to read this";
          i.classList.remove("tip"); root.append(i);
        }
        for (const e of model.entries) {
          // A label, so a click anywhere on the entry ticks its box.
          const s = item("entry", e.about, e.toggle ? "label" : "span");
          if (e.toggle) {
            const box = document.createElement("input"); box.type = "checkbox";
            const on = !model.hidden.includes(e.toggle);
            box.checked = on; s.classList.toggle("off", !on);
            box.addEventListener("change", () => {
              const rest = model.hidden.filter(h => h !== e.toggle);
              model.hidden = box.checked ? rest : [...rest, e.toggle];
            });
            s.append(box);
          }
          const m = document.createElement("span"); m.className = "mark";
          m.style.color = e.color; m.textContent = e.mark;
          s.append(m, document.createTextNode(e.label));
          root.append(s);
        }
      }
      for (const name of ["title", "about", "entries", "hidden"]) model.on(name, draw);
      model.on("remove", hide);
      window.addEventListener("scroll", hide, true);
      draw();
      return root;
    }
    """


def key(title: str, about: str, entries: list[dict] = (), **kwargs) -> Key:
    """The header of a plot titled ``title``; ``about`` is plain text, paragraphs apart by a blank
    line."""
    return Key(
        title=title,
        about=_html(about),
        entries=list(entries),
        sizing_mode="stretch_width",
        margin=(0, 10),
        **kwargs,
    )
