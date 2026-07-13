# Ship I — Streamlit polish guide (`app.py`)

**Status:** planned (part of Ship I). You build; I review against the `qa` seam
before the screenshot.
**Goal:** turn the bare Ship D skeleton (`app.py`, ~71 lines) into a demo that
screenshots well for the portfolio — looks intentional, guides a reviewer who
doesn't know what to ask, and visibly reflects the "grounded + cited + evaluated"
story.

**Scope (locked):** all seven items below (the 4 high-value + sidebar About +
`st.form` + `Settings()` dedup).

The `qa` seam this renders: `QAEngine.answer_query(query, top_k) -> GroundedAnswer`
with `.answer` (str, inline `[n]` markers), `.citations` (list of `Citation`:
`marker, chunk_id, article_id, title, source, url`), `.answered_from_context` (bool).

---

## 1. `page_config` — must be first

```python
st.set_page_config(page_title="FinNews-RAG", page_icon="📈", layout="wide")
```

**Gotcha:** must be the **first Streamlit call in the script run** — before
`st.title`, before anything renders. Put it as the first line of `main()`.
`get_engine()` is fine after it (it doesn't render). Any `st.*` before it →
`set_page_config() can only be called once and must be the first`.

## 2. Example-query buttons — *the fiddly one*

Pattern: give the input a **key**, and have example buttons write that key in
`session_state` **before the input widget is created**.

- Give the text field `key="query_box"`.
- Render the example `st.button(...)`s **above** the input. On click, set
  `st.session_state["query_box"] = "What did the Fed signal on rates?"`.
- **Gotcha:** you can only assign a widget-backed `session_state` key *before*
  that widget is instantiated in the run. A button click triggers a rerun, so the
  button code (above) runs first, sets the value, then the input below picks it
  up. Setting it *after* the `text_input` line raises
  `cannot be modified after the widget ... is instantiated`.
- **Gotcha with the form (item 6):** plain `st.button`s **cannot live inside
  `st.form`** (only `st.form_submit_button` can). So example buttons go
  *outside/above* the form. They pre-fill the box; the user still submits.

## 3. Source cards + clickable links

```python
with st.container(border=True):   # needs Streamlit >= 1.29 (you have >=1.37)
    st.markdown(f"**[{c.title}]({c.url})**  \n{c.source} · [{c.marker}]")
```

Markdown `[text](url)` renders a real hyperlink. Optional: collapse duplicate
citations on `article_id` if the same article appears under two markers — skippable.

## 4. Abstention banner

Branch **before** rendering the answer:

```python
if not result.answered_from_context:
    st.warning("Not enough indexed context to answer that confidently.")
```

Decide whether to still show `result.answer` underneath (the seam returns a polite
"I don't have enough indexed context…" string for the zero-hit case). For the
screenshot, banner + that line reads as "refuses to hallucinate" — worth
capturing the out-of-domain case *well*.

## 5. Sidebar About

```python
with st.sidebar:
    st.header("About")
    st.markdown("...")
```

Content to tie to the README: grounded RAG over financial news · **custom** eval
harness (faithfulness 0.96, answer-relevance 0.82) · no LangChain/LlamaIndex.
This is where a reviewer's eye goes after the answer.

## 6. `st.form`

```python
with st.form("ask"):
    query = st.text_input("Ask about recent financial news", key="query_box")
    submitted = st.form_submit_button("Ask")
```

**Gotcha:** inside a form nothing reruns until submit — read `submitted` (or
`st.session_state["query_box"]`) after the block, and drop the old standalone
`st.button("Ask")`. Enter-in-field submits the form for free.

## 7. `Settings()` dedup

Currently built twice (in `get_engine()` and `main()`). Cleanest fix — a cached
accessor:

```python
@st.cache_resource
def get_settings() -> Settings:
    return Settings()  # type: ignore
```

Call it in both places; `main()` reads `get_settings().RETRIEVAL_TOP_K`.

---

## Ordering in `main()` that avoids every gotcha

`set_page_config` → sidebar → example buttons (write `session_state`) → `st.form`
(input keyed `query_box` + submit) → resolve the query → spinner/`answer_query` →
abstention banner → answer → source cards.

Then `streamlit run app.py`, review, screenshot for the README.
